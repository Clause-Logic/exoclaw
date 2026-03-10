# Running exoclaw in Temporal

This document captures how to run exoclaw inside a [Temporal](https://temporal.io) workflow engine.

## Why it fits

The `AgentLoop` turn is already shaped like a Temporal workflow — pure orchestration with no business logic of its own:

```
build_prompt → call LLM → [execute tools]* → record → reply
```

The five exoclaw protocols map cleanly to Temporal primitives. The `Executor` protocol in core provides the seam — implement it for your execution environment and everything else stays the same.

---

## What survives unchanged

- `LLMProvider`, `Conversation`, `Tool`, `Channel`, `Bus` — **all untouched**. No protocol changes, no signature changes.
- Hook callbacks (`on_pre_context`, `on_pre_tool`, `on_post_turn`, `on_max_iterations`) — already shaped like activities. No changes.
- `system_context()` on tools — sync, called in workflow code before `build_prompt`. No changes.
- Conversation JSONL storage — unchanged, just called via an activity.
- Subagents (`exoclaw-tools-spawn`) — map to Temporal child workflows (see below).

---

## The Executor protocol

The `Executor` protocol is the only new protocol in core. It has one method per I/O operation, giving the implementation full control over how each operation is executed (timeout policies, retry strategies, task queue routing):

```python
class Executor(Protocol):
    async def chat(self, provider, *, messages, tools, model, temperature, max_tokens, reasoning_effort) -> LLMResponse: ...
    async def execute_tool(self, registry, name, params, ctx) -> str: ...
    async def build_prompt(self, conversation, session_id, message, **kwargs) -> list[dict]: ...
    async def record(self, conversation, session_id, new_messages) -> None: ...
    async def clear(self, conversation, session_id) -> bool: ...
    async def run_hook(self, fn, /, *args, **kwargs) -> Any: ...
```

The default `DirectExecutor` calls everything inline — zero behavior change for existing users. Pass `executor=` to `AgentLoop` or `Exoclaw` to swap it.

A Temporal implementation would look like:

```python
class TemporalExecutor:
    async def chat(self, provider, **kwargs):
        return await workflow.execute_activity(
            llm_chat_activity, kwargs,
            start_to_close_timeout=timedelta(minutes=5),
        )

    async def execute_tool(self, registry, name, params, ctx):
        return await workflow.execute_activity(
            tool_activity, (name, params, ctx),
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=30),
        )

    async def build_prompt(self, conversation, session_id, message, **kwargs):
        return await workflow.execute_activity(
            build_prompt_activity, (session_id, message, kwargs),
            start_to_close_timeout=timedelta(seconds=30),
        )

    async def record(self, conversation, session_id, new_messages):
        await workflow.execute_activity(
            record_activity, (session_id, new_messages),
            start_to_close_timeout=timedelta(seconds=30),
        )

    async def clear(self, conversation, session_id):
        return await workflow.execute_activity(
            clear_activity, session_id,
            start_to_close_timeout=timedelta(seconds=30),
        )

    async def run_hook(self, fn, /, *args, **kwargs):
        return await workflow.execute_local_activity(
            hook_activity, (fn, args, kwargs),
            start_to_close_timeout=timedelta(seconds=10),
        )
```

Each method gets its own timeout, retry policy, and task queue routing. `chat` gets a long timeout; `execute_tool` gets heartbeating for long-running tools; `run_hook` uses local activities for low-latency callbacks.

---

## Dispatch model

The existing `process_direct()` method on `AgentLoop` already provides a coupled request-response interface:

```python
result = await loop.process_direct("do something", session_key="s:1")
```

A Temporal workflow can call this through an activity directly. No new `Dispatcher` protocol is needed in core — the dispatch model is a Temporal-layer concern:

```python
@workflow.defn
class AgentTurnWorkflow:
    @workflow.run
    async def run(self, input: TurnInput) -> str:
        return await workflow.execute_activity(
            process_turn_activity, input,
            start_to_close_timeout=timedelta(minutes=15),
        )
```

For long-running session workflows with Updates, the Temporal package provides its own dispatcher that wraps `process_direct()`:

```python
@workflow.defn
class SessionWorkflow:
    @workflow.update
    async def send_message(self, msg: InboundMessage) -> str:
        return await workflow.execute_activity(
            process_turn_activity, msg,
            start_to_close_timeout=timedelta(minutes=15),
        )

    @workflow.run
    async def run(self, session_id: str) -> None:
        await workflow.wait_condition(lambda: False, timeout=timedelta(days=365))
```

Channels that want to talk to a Temporal-backed agent use the Temporal client SDK to send Updates. Channels that don't want Temporal keep using the Bus as before. No protocol changes needed.

---

## What goes away (in Temporal mode)

These components are replaced by Temporal equivalents in the Temporal package, but remain available for non-Temporal deployments:

| current | Temporal replacement |
|---|---|
| Outer `while self._running` loop in `AgentLoop.run()` | Temporal dispatches to the workflow |
| `MessageBus` at the channel boundary | Temporal Updates |
| Cron JSON file (`exoclaw-tools-cron` storage) | Temporal cron child workflows |
| `_active_tasks` dict + `cancel_by_session` | Temporal child workflow cancellation handles |
| `asyncio.Lock`, `asyncio.create_task`, `asyncio.ensure_future` in the loop | Temporal's own scheduler |

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
| `Executor` protocol + `DirectExecutor` | core (exoclaw) — **already landed** |
| `AgentLoop` accepts optional `executor: Executor` | core (exoclaw) — **already landed** |
| `Exoclaw` accepts optional `executor: Executor` | core (exoclaw) — **already landed** |
| `TemporalExecutor` implementation | temporal package (external) |
| Workflow definitions (`AgentTurnWorkflow`, `SessionWorkflow`) | temporal package (external) |
| Activity wrappers for LLM, tools, conversation | temporal package (external) |

**No existing protocols, channel implementations, tool implementations, or provider implementations need to change.** The Executor is opt-in — pass one to get a different execution environment, or don't and everything works as before.
