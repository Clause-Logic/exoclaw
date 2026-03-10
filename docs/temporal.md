# Running exoclaw in Temporal

This document captures what would need to change to run exoclaw inside a [Temporal](https://temporal.io) workflow engine, based on a design exploration against the openclaw reference implementation.

## Why it fits

The `AgentLoop` turn is already shaped like a Temporal workflow — pure orchestration with no business logic of its own:

```
build_prompt → call LLM → [execute tools]* → record → reply
```

The five exoclaw protocols map cleanly to Temporal primitives. The main work is adding two new protocols and replacing one signature.

---

## What survives unchanged

- `LLMProvider`, `Conversation`, `Tool` — untouched. Called from Temporal activities instead of directly from the loop, but implementations don't change.
- `Channel` protocol shape (`start`, `stop`, `send`) — shape stays, signature of `start` changes (see below).
- `Bus` — stays but becomes internal-only. The loop still uses it internally; channels no longer touch it.
- Hook callbacks (`on_pre_context`, `on_pre_tool`, `on_post_turn`, `on_max_iterations`) — already shaped like activities. No changes.
- `system_context()` on tools — sync, called in workflow code before `build_prompt`. No changes.
- Conversation JSONL storage — unchanged, just called via an activity.
- Subagents (`exoclaw-tools-spawn`) — map to Temporal child workflows (see below).

---

## Two new protocols

### 1. `Executor`

The seam between "call this directly" and "call this as a Temporal activity". Makes the loop environment-agnostic.

```python
class Executor(Protocol):
    async def run(self, fn: Callable, /, *args: Any, **kwargs: Any) -> Any: ...
```

Two implementations:

```python
class DirectExecutor:
    async def run(self, fn, /, *args, **kwargs):
        return await fn(*args, **kwargs)

class TemporalExecutor:
    async def run(self, fn, /, *args, **kwargs):
        return await workflow.execute_activity(fn, *args, **kwargs)
```

The loop replaces every I/O `await` with `self.executor.run(...)`:

| current | via executor |
|---|---|
| `await self.provider.chat(...)` | `await self.executor.run(self.provider.chat, ...)` |
| `await self.conversation.build_prompt(...)` | `await self.executor.run(self.conversation.build_prompt, ...)` |
| `await self.tools.execute(...)` | `await self.executor.run(self.tools.execute, ...)` |
| `await self.conversation.record(...)` | `await self.executor.run(self.conversation.record, ...)` |
| `await self._on_pre_context(...)` | `await self.executor.run(self._on_pre_context, ...)` |
| `await self._on_pre_tool(...)` | `await self.executor.run(self._on_pre_tool, ...)` |

Same loop, same protocols, same implementations. Swap the executor and you're in a different runtime.

---

### 2. `Dispatcher`

The current channel model is decoupled: a channel publishes inbound via the bus, then the loop separately consumes outbound and calls `channel.send()`. That's a queue model.

Temporal Updates are coupled — you submit a message and get the response back on the same call. A `Dispatcher` protocol abstracts this:

```python
class Dispatcher(Protocol):
    async def dispatch(self, msg: InboundMessage) -> str: ...
```

Two implementations:

```python
class BusDispatcher:
    """Direct mode: publish to bus, wait for outbound response."""
    async def dispatch(self, msg: InboundMessage) -> str:
        await self._bus.publish_inbound(msg)
        response = await self._bus.consume_outbound()
        return response.content

class TemporalDispatcher:
    """Temporal mode: send an Update, get the response back directly."""
    async def dispatch(self, msg: InboundMessage) -> str:
        return await self._workflow_handle.execute_update(
            SessionWorkflow.send_message, msg
        )
```

`Channel.start(bus: Bus)` becomes `Channel.start(dispatcher: Dispatcher)`. The channel calls `dispatcher.dispatch(msg)` instead of `bus.publish_inbound(msg)`, and never needs to consume outbound itself — the Dispatcher owns that.

---

## One signature change

```python
# before
async def start(self, bus: Bus) -> None: ...

# after
async def start(self, dispatcher: Dispatcher) -> None: ...
```

Channel implementations (Telegram, IPC, etc.) change the argument they receive and call `dispatcher.dispatch(msg)` instead of `bus.publish_inbound(msg)`. Everything else in the channel stays the same.

---

## What goes away

| current | replacement |
|---|---|
| Outer `while self._running` loop in `AgentLoop.run()` | Temporal dispatches to the workflow |
| `MessageBus` at the channel boundary | `Dispatcher` protocol |
| Cron JSON file (`exoclaw-tools-cron` storage) | Temporal cron child workflows |
| `_active_tasks` dict + `cancel_by_session` | Temporal child workflow cancellation handles |
| `asyncio.Lock`, `asyncio.create_task`, `asyncio.ensure_future` in the loop | Temporal's own scheduler |

---

## Workflow structure

### Per-turn (simpler)

Each inbound message starts a new workflow. Session state lives entirely in the `Conversation` implementation (external storage).

```python
@workflow.defn
class AgentTurnWorkflow:
    @workflow.run
    async def run(self, input: TurnInput) -> str:
        messages = await workflow.execute_activity(build_prompt, input)
        response = await workflow.execute_activity(llm_chat, messages)

        while response.has_tool_calls:
            for call in response.tool_calls:
                result = await workflow.execute_activity(execute_tool, call)
                messages = append_tool_result(messages, call, result)
            response = await workflow.execute_activity(llm_chat, messages)

        await workflow.execute_activity(record_turn, messages)
        return response.content
```

### Long-running session (more powerful)

One workflow per session, receives messages via Updates. Temporal's event history is the source of truth for in-flight state.

```python
@workflow.defn
class SessionWorkflow:
    def __init__(self):
        self._queue: list[InboundMessage] = []

    @workflow.update
    async def send_message(self, msg: InboundMessage) -> str:
        # Run the turn, return the response
        return await self._process_turn(msg)

    @workflow.run
    async def run(self, session_id: str) -> None:
        # Long-running; terminated explicitly when session ends
        await workflow.wait_condition(lambda: False, timeout=timedelta(days=365))
```

The `send_message` Update is what the `TemporalDispatcher` calls. The caller blocks until the turn completes and gets the response back directly.

---

## Subagents as child workflows

`exoclaw-tools-spawn` creates asyncio tasks today. In Temporal these become child workflows:

| exoclaw | Temporal |
|---|---|
| `asyncio.create_task(loop.run())` | `workflow.start_child_workflow(AgentTurnWorkflow, ...)` |
| fire-and-forget | detached child workflow |
| `cancel_by_session()` | child workflow cancellation handle |
| session key | child workflow ID |

Temporal child workflows survive worker crashes independently, which is stronger than the current asyncio task model.

---

## Shell sandboxing

The shell execution tool becomes a Temporal activity. The sandbox lifecycle maps to a child workflow:

```
SessionWorkflow
  └── SandboxWorkflow (child, owns container lifetime)
        ├── activity: create_container()
        ├── activity: exec_command()   ← each shell tool call
        ├── activity: exec_command()
        └── activity: destroy_container()
```

Options for the sandbox itself:
- **E2B** — purpose-built cloud sandboxes for AI agents, Python SDK, persistent per session
- **Docker SDK** — spin up a container per session, exec for each shell call
- **Firecracker microVMs** — VM-level isolation if container escape is a concern
- **gVisor** — stronger isolation than standard Docker, drop-in via `--runtime=runsc`

The sandbox is guaranteed to be cleaned up when the session workflow ends, even on worker crash.

---

## Summary

| change | scope |
|---|---|
| New `Executor` protocol | core (exoclaw) |
| New `Dispatcher` protocol | core (exoclaw) |
| `Channel.start(bus)` → `Channel.start(dispatcher)` | protocol + all channel impls |
| `AgentLoop` accepts `executor: Executor` | core (exoclaw) |
| Loop `await x()` → `await self.executor.run(x, ...)` | core (exoclaw) |
| Channel impls call `dispatcher.dispatch()` not `bus.publish_inbound()` | all channel packages |

Everything else — all five existing protocols, all plugin implementations, all storage — survives without changes.
