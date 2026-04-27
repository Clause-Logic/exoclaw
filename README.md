# exoclaw 🦀

[![PyPI](https://img.shields.io/pypi/v/exoclaw)](https://pypi.org/project/exoclaw/)
[![CI](https://github.com/Clause-Logic/exoclaw/actions/workflows/pr.yml/badge.svg)](https://github.com/Clause-Logic/exoclaw/actions/workflows/pr.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI agent infrastructure that fits in your stack, not the other way around.**

You have an app. Wire in exoclaw and it becomes intelligent — tool use, session memory, multi-turn conversations, any LLM. You own every piece. Nothing is baked in.

```
pip install exoclaw
```

One runtime dependency: `structlog`.

> **Want a running bot in 30 seconds?** Skip ahead to [`exoclaw-nanobot`](https://github.com/Clause-Logic/exoclaw-plugins/tree/main/packages/exoclaw-nanobot) — the one-line bundle that wires every plugin together and gives you a working agent. The full plugin index lives at [`exoclaw-plugins`](https://github.com/Clause-Logic/exoclaw-plugins) (provider, conversation, tools, channels, executors). This repo is the protocol-only core that all of those build on; read on if you want to know what's underneath.

---

## Origin

exoclaw is a fork of [nanobot](https://github.com/NanobotAI/nanobot), stripped down to ~2,000 lines of auditable Python.

That's it. Read it in an afternoon. Understand exactly what you're shipping. Then wire it into your FastAPI app, your [GitHub Actions workflow](https://github.com/Clause-Logic/exoclaw-github), your Slack bot, your CLI — whatever you're building. Your stack gains OpenClaw-grade agentic capabilities without taking on a framework as a dependency.

The original nanobot ships with batteries — LLM provider, memory system, cron, MCP, Telegram, Discord. Convenient to start. But every baked-in feature is a PR waiting to happen. A Telegram API change breaks a cron bug fix release. An MCP upgrade pulls in conflicts for users who don't use MCP. The framework and its features are entangled.

exoclaw cuts the knot. Five protocols, one loop, ~2,000 lines. Everything else — storage, channels, tools, providers — lives in separate packages you opt into. The core never changes because it has nothing to change.

- **Auditable.** ~2,000 lines, mypy strict, 95% test coverage. You can read and understand it in an afternoon.
- **No dependency drag.** Your tree contains exactly what you chose.
- **No surprise breakage.** A bug in someone else's Telegram plugin can't break your app.
- **Composable.** Swap providers, storage, or channels without touching the loop.

---

## How it works

exoclaw is seven protocols and a loop.

```
InboundMessage → Bus → AgentLoop → LLM → Tools → Bus → OutboundMessage → Channel
```

1. A **Channel** receives a message from the outside world and puts it on the **Bus**
2. The **AgentLoop** pulls it off the bus, asks the **Conversation** to build a prompt
3. The prompt goes to the **LLMProvider**, which returns a response
4. If the response has tool calls, the loop executes them via registered **Tools**
5. The final response goes back on the bus, and the **Channel** delivers it

Every one of those nouns is a protocol. Swap any of them out. No inheritance required.

| Protocol | You implement | Default provided | Notes |
|---|---|---|---|
| `LLMProvider` | **yes** | — | Use a plugin like `exoclaw-provider-litellm` |
| `Conversation` | **yes** | — | Use a plugin like `exoclaw-conversation` |
| `Tool` | optional | — | Pass tools you need, or none |
| `Channel` | optional | — | Pass channels you need, or use `process_direct()` |
| `Bus` | optional | `MessageBus` | Asyncio queues, sufficient for single-process |
| `Executor` | optional | `DirectExecutor` | Inline execution, zero overhead |
| `IterationPolicy` | optional | — | Hard `max_iterations` cap when absent |

---

## The Protocols

### `LLMProvider`

```python
class LLMProvider(Protocol):
    def get_default_model(self) -> str: ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None,
    ) -> LLMResponse: ...
```

`LLMResponse` carries `.content`, `.tool_calls`, `.finish_reason`, `.has_tool_calls`.

**Plugin ideas:**
- `exoclaw-provider-litellm` — route to any model via LiteLLM
- `exoclaw-provider-anthropic` — direct Anthropic SDK
- `exoclaw-provider-openai` — direct OpenAI SDK
- `exoclaw-provider-ollama` — local models

---

### `Conversation`

```python
class Conversation(Protocol):
    async def build_prompt(
        self,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def record(self, session_id: str, new_messages: list[dict[str, Any]]) -> None: ...
    async def clear(self, session_id: str) -> bool: ...
    def list_sessions(self) -> list[dict[str, Any]]: ...
```

`build_prompt` returns the full message list sent to the LLM — system prompt, history, new user message. `plugin_context` strings are collected from tools that implement `system_context()` and injected into the system prompt.

**Plugin ideas:**
- `exoclaw-conversation` — file-backed sessions, JSONL history, LLM memory consolidation
- `exoclaw-conversation-redis` — Redis-backed for multi-instance deployments
- `exoclaw-conversation-postgres` — durable storage with vector memory

---

### `Tool`

```python
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    async def execute(self, **kwargs: Any) -> str: ...
```

Tools are registered at construction time via `Exoclaw(tools=[...])`. The loop calls `tool.execute(**args)` and feeds results back into the LLM context.

**Optional hooks** (duck-typed — implement if you need them):

```python
def on_inbound(self, msg: InboundMessage) -> None:
    """Called before each message is processed. Update per-turn state here."""

def system_context(self) -> str:
    """Return a string injected into the system prompt every turn."""

async def execute_with_context(self, ctx: ToolContext, **kwargs: Any) -> str:
    """Like execute(), but receives ToolContext(session_key, channel, chat_id).
    The registry calls this instead of execute() when context is available."""

def set_bus(self, bus: Bus) -> None:
    """Called at registration time. Lets tools publish back to the bus
    for async/background work that re-enters the loop later."""

async def cancel_by_session(self, session_key: str) -> int:
    """Cancel running work for a session. Return count cancelled. Called on /stop."""

sent_in_turn: bool  # If True after execute(), loop suppresses the normal reply
```

**Loop lifecycle callbacks** (pass to `AgentLoop.__init__` — all optional):

```python
AgentLoop(
    ...,
    # Called before build_prompt. Return value is appended to the system prompt.
    on_pre_context=async def(message, session_key, channel, chat_id) -> str,

    # Called before each tool execution. Return a non-empty string to reject
    # the call — the string is fed back to the LLM as the tool result.
    on_pre_tool=async def(tool_name, tool_args, session_key) -> str | None,

    # Called after each turn is recorded. Fire-and-forget.
    on_post_turn=async def(messages, session_key, channel, chat_id) -> None,

    # Called when the tool call iteration limit is reached. Fire-and-forget.
    on_max_iterations=async def(session_key, channel, chat_id) -> None,
)
```

**Plugin ideas:**
- `exoclaw-tools-mcp` — connect MCP servers, register each as a Tool
- `exoclaw-tools-web` — web search and page fetching
- `exoclaw-tools-shell` — sandboxed shell execution
- `exoclaw-tools-files` — workspace file operations
- `exoclaw-tools-memory` — read/write long-term memory files
- `exoclaw-tools-message` — send messages to other channels (sets `sent_in_turn=True`)
- `exoclaw-tools-cron` — schedule reminders (implements `system_context()` + `on_inbound()`)
- `exoclaw-tools-skills` — load SKILL.md files from a workspace directory and inject via `system_context()`

---

### `Channel`

```python
class Channel(Protocol):
    name: str

    async def start(self, bus: Bus) -> None:
        """Connect to the platform and begin receiving messages."""

    async def stop(self) -> None:
        """Disconnect and release resources."""

    async def send(self, msg: OutboundMessage) -> None:
        """Deliver an outbound message to the platform."""
```

The bus is injected at `start()` time — channels are constructed without it, so synthetic channels (heartbeat, cron triggers) can be created before the bus exists.

**Plugin ideas:**
- `exoclaw-channel-telegram` — Telegram bot
- `exoclaw-channel-discord` — Discord bot
- `exoclaw-channel-slack` — Slack app
- `exoclaw-channel-cli` — interactive terminal REPL
- `exoclaw-channel-heartbeat` — timed pings that trigger background agent tasks
- `exoclaw-channel-cron` — cron-scheduled messages routed to the agent

---

### `Bus`

```python
class Bus(Protocol):
    async def publish_inbound(self, msg: InboundMessage) -> None: ...
    async def consume_inbound(self) -> InboundMessage: ...
    async def publish_outbound(self, msg: OutboundMessage) -> None: ...
    async def consume_outbound(self) -> OutboundMessage: ...
```

The default `MessageBus` is a pair of asyncio queues — sufficient for single-process deployments.

**Plugin ideas:**
- `exoclaw-bus-redis` — Redis pub/sub for multi-process or distributed agents
- `exoclaw-bus-nats` — NATS for high-throughput pipelines

---

### `Executor`

```python
class Executor(Protocol):
    async def chat(self, provider, *, messages, tools, model, temperature, max_tokens, reasoning_effort) -> LLMResponse: ...
    async def execute_tool(self, registry, name, params, ctx) -> str: ...
    async def build_prompt(self, conversation, session_id, message, **kwargs) -> list[dict]: ...
    async def record(self, conversation, session_id, new_messages) -> None: ...
    async def clear(self, conversation, session_id) -> bool: ...
    async def run_hook(self, fn, /, *args, **kwargs) -> Any: ...
    async def run_turn(self, loop, session_id, message, **kwargs) -> tuple | None: ...
    async def mint_turn_id(self) -> str: ...
```

The `Executor` controls how the agent loop performs I/O. One method per operation, so each can have its own execution strategy.

The default `DirectExecutor` calls everything inline — zero overhead, identical to the behavior before the protocol existed. Pass `executor=` to `AgentLoop` or `Exoclaw` to swap it:

```python
from exoclaw import Exoclaw

app = Exoclaw(
    provider=provider,
    conversation=conversation,
    executor=my_custom_executor,  # opt-in
)
```

This is how you run exoclaw in different execution environments (workflow engines, distributed task queues, etc.) without changing any other protocol, channel, tool, or provider implementation.

**Plugin ideas:**
- `exoclaw-executor-temporal` — run each operation as a Temporal activity with per-operation timeouts and retry policies
- `exoclaw-executor-celery` — route tool execution through Celery workers

#### Turn trace context

Every call to `AgentLoop._process_turn_inline` mints a `turn.id` via `executor.mint_turn_id()` and binds a trace context into [structlog's contextvars](https://www.structlog.org/en/stable/contextvars.html) for the duration of the turn:

| Field | Meaning |
|---|---|
| `turn.id` | This turn's unique id (uuidv7, time-ordered) |
| `turn.root_id` | The originating turn — immutable across the whole subagent tree |
| `turn.parent_id` | The direct parent turn, or `None` at the top |
| `turn.depth` | `0` for a top-level turn, `1` for a subagent of a top-level, etc. |
| `turn.chain` | Full ancestry as a `:`-joined string, e.g. `root:child:self` |

Every downstream log line — LLM requests, tool calls, tool results, subagent spawns — inherits these fields automatically. That gives you one-query debugging: `turn.root_id:<uuid>` in your log backend surfaces every line from a single user message, including every downstream subagent turn, with no joins or timestamp correlation.

Nested `_process_turn_inline` calls (the pattern a subagent uses when it re-enters the loop inside the same process) read the existing `turn.*` contextvars before binding, so the chain extends correctly instead of being wiped.

Durable executors (DBOS, Temporal) implement `mint_turn_id` by wrapping a non-deterministic source in whatever their framework provides for replay-safe side-effects — `@DBOS.step()` for DBOS, `workflow.side_effect` for Temporal — so the same id is returned on workflow recovery. `DirectExecutor` has no replay boundary to worry about and just returns a fresh uuidv7 inline.

---

### `IterationPolicy`

```python
class IterationPolicy(Protocol):
    async def should_continue(self, iteration: int, tools_used: list[str]) -> bool: ...
    async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str: ...
```

Controls when the agent loop stops iterating. Without a policy, the loop uses a hard `max_iterations` counter (default 40). With a policy, you can implement smarter termination — pattern detection, adaptive budgets, or anything else.

The policy is **orthogonal to the executor**. A Temporal executor handles *how* operations run; an iteration policy handles *when to stop*. Compose them independently:

```python
from exoclaw import Exoclaw

app = Exoclaw(
    provider=provider,
    conversation=conversation,
    executor=temporal_executor,           # how to run operations
    iteration_policy=loop_detection,      # when to stop iterating
)
```

**Plugin ideas:**
- `exoclaw-loop-detection` — openclaw-style pattern detection (repeat, ping-pong, circuit breaker) instead of a hard cap

---

## Usage

### GitHub Actions bot

The quickest way to run exoclaw in production — zero infrastructure, no servers. See [exoclaw-github](https://github.com/Clause-Logic/exoclaw-github) for the full plugin and a [live demo](https://github.com/Clause-Logic/exoclaw-github-demo).

Drop a workflow file in your repo and the bot responds to issues and PR comments using your `GITHUB_TOKEN` — no extra secrets needed:

```yaml
# .github/workflows/bot.yml
- uses: Clause-Logic/exoclaw-github@main
  with:
    trigger: "@exoclawbot"
    tools: github_pr_diff, github_file, github_checks, github_review, github_label
```

---

### Minimal agent in Python

```python
import asyncio
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.bus.events import InboundMessage

# Plugin packages — not part of exoclaw core
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_conversation.conversation import DefaultConversation

async def main():
    bus = MessageBus()
    provider = LiteLLMProvider(default_model="claude-sonnet-4-6")
    conversation = DefaultConversation.create(
        workspace="~/.mybot",
        provider=provider,
        model="claude-sonnet-4-6",
    )

    loop = AgentLoop(bus=bus, provider=provider, conversation=conversation)
    asyncio.create_task(loop.run())

    # Publish a message and consume the response
    await bus.publish_inbound(InboundMessage(
        channel="cli", sender_id="user", chat_id="main", content="Hello!"
    ))
    response = await bus.consume_outbound()
    print(response.content)

asyncio.run(main())
```

---

### Drop into an existing web app

exoclaw doesn't own your event loop — wire the bus into whatever you already have. Here's FastAPI:

```python
from fastapi import FastAPI
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.bus.events import InboundMessage, OutboundMessage

app = FastAPI()
bus = MessageBus()
agent = AgentLoop(bus=bus, provider=..., conversation=...)

@app.on_event("startup")
async def start_agent():
    import asyncio
    asyncio.create_task(agent.run())

@app.post("/chat")
async def chat(user_id: str, message: str):
    await bus.publish_inbound(InboundMessage(
        channel="api", sender_id=user_id, chat_id=user_id, content=message,
    ))
    response: OutboundMessage = await bus.consume_outbound()
    return {"reply": response.content}
```

The agent loop runs as a background task. Your API routes are just producers and consumers on the bus.

---

### Durable execution with Temporal

exoclaw's `Executor` protocol is the hook that enables running agents on [Temporal](https://temporal.io) without changing any tool, channel, or provider.

[exoclaw-temporal](https://github.com/Clause-Logic/exoclaw-temporal) implements `AgentTurnWorkflow` — the agent loop rewritten as a Temporal workflow where each operation is a Temporal activity:

| Executor method | Temporal activity | What it means |
|---|---|---|
| `build_prompt` | `build_prompt_activity` | Load history from shared volume |
| `chat` | `llm_chat_activity` | LLM call with retry on transient failure |
| `execute_tool` | `execute_tool_activity` | Tool call with heartbeat — survives worker death |
| `record` | `record_turn_activity` | Persist new messages to shared volume |

The result: every tool call, every LLM call, every retry is checkpointed. If a worker pod dies mid-execution, Temporal reschedules on a survivor. The agent resumes exactly where it left off — not from the start of the turn, but from the exact activity that was interrupted.

```python
from exoclaw_temporal.config import LLMConfig, TurnInput, WorkspaceConfig
from exoclaw_temporal.turn_based.workflows.agent_turn import AgentTurnWorkflow
from temporalio.client import Client

client = await Client.connect("localhost:7233")
result = await client.execute_workflow(
    AgentTurnWorkflow.run,
    TurnInput(
        session_id="my-session",
        message="Write a summary of this codebase.",
        llm=LLMConfig(model="anthropic/claude-sonnet-4-6"),
        workspace=WorkspaceConfig(path="/workspace"),
        ...
    ),
    id="turn-1",
    task_queue="exoclaw-temporal",
)
print(result.final_content)
```

See [exoclaw-temporal](https://github.com/Clause-Logic/exoclaw-temporal) for the full setup, Kubernetes deployment, bounce demo, and session-based approach (one long-running workflow per conversation).

---

### Swap components without touching the loop

```python
# File-backed sessions (default)
from exoclaw_conversation.conversation import DefaultConversation
conversation = DefaultConversation.create(workspace="~/.mybot", ...)

# → swap for Redis without changing anything else
from exoclaw_conversation_redis import RedisConversation
conversation = RedisConversation(url="redis://localhost", ...)

# Local model
provider = LiteLLMProvider(default_model="ollama/llama3")

# → swap for Anthropic without changing anything else
provider = LiteLLMProvider(default_model="claude-sonnet-4-6")

# Same AgentLoop, same Bus, same tools — only the component changed
loop = AgentLoop(bus=bus, provider=provider, conversation=conversation, tools=[...])
```

---

## Writing a Tool

```python
from exoclaw.agent.tools.protocol import ToolBase  # optional mixin

class WeatherTool(ToolBase):
    name = "get_weather"
    description = "Get the current weather for a city."
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    }

    async def execute(self, city: str) -> str:
        # fetch weather...
        return f"It's sunny in {city}, 22°C."

    def system_context(self) -> str:
        return "You have access to real-time weather data via get_weather."
```

No base class required — `ToolBase` is an optional mixin that gives you parameter casting, validation, and schema generation for free. Implement the four attributes and `execute()` directly if you prefer.

---

## Writing a Channel

```python
from exoclaw.bus.events import InboundMessage, OutboundMessage
from exoclaw.bus.protocol import Bus

class WebhookChannel:
    name = "webhook"

    async def start(self, bus: Bus) -> None:
        self._bus = bus
        # start your web server, register routes, etc.

    async def stop(self) -> None:
        # shut down web server
        pass

    async def send(self, msg: OutboundMessage) -> None:
        # deliver msg.content to the webhook target
        pass

    async def _on_request(self, payload: dict) -> None:
        await self._bus.publish_inbound(InboundMessage(
            channel=self.name,
            sender_id=payload["user_id"],
            chat_id=payload["chat_id"],
            content=payload["text"],
        ))
```

---

## Plugin system

Tools and channels can inject context into the system prompt each turn via `system_context()`:

```python
class CronTool:
    name = "cron"
    # ...

    def system_context(self) -> str:
        jobs = self._list_active_jobs()
        return f"# Scheduled Jobs\n\n{jobs}"
```

The loop collects `system_context()` from all registered tools before each `build_prompt` call and passes the results as `plugin_context`. Each plugin owns its own section of the system prompt — no static template files needed.

---

## Project structure

```
exoclaw/
  app.py                   # Exoclaw — the composition root
  executor.py              # Executor protocol + DirectExecutor
  iteration_policy.py      # IterationPolicy protocol
  agent/
    loop.py                # AgentLoop — the core processing engine
    conversation.py        # Conversation protocol
    tools/
      protocol.py          # Tool protocol + ToolBase mixin
      registry.py          # ToolRegistry
  bus/
    protocol.py            # Bus protocol
    events.py              # InboundMessage, OutboundMessage
    queue.py               # Default asyncio queue implementation
  channels/
    protocol.py            # Channel protocol
    manager.py             # ChannelManager
  providers/
    protocol.py            # LLMProvider protocol
    types.py               # LLMResponse, ToolCallRequest
```

---

## MicroPython support

exoclaw core runs on **CPython and MicroPython** out of the same source tree. The compat shim in `exoclaw/_compat.py` papers over runtime differences (`ContextVar`, `inspect.iscoroutinefunction`, `secrets.token_bytes`, `tempfile.mkstemp`, `threading.Lock`, `structlog`); call sites import the shim and don't branch on the runtime.

**It's safe to use because CI proves it.** Every PR runs the test suite as a matrix on both runtimes — CPython `pytest --cov=exoclaw` and a MicroPython runner (`tests/_micropython_runner/run.py`) that executes the same tests under MP's unix port and traces line coverage via `sys.settrace`. Both must independently hit ≥95% on their own reachable line set, with `# pragma: no cover (cpython|micropython)` markers gating per-runtime branches. A change that breaks MicroPython compat fails the matrix even if CPython stays green.

The aspirational target is detailed in `docs/memory-model.md` — the same memory architecture work that bounded multi-tenant openclaw's RAM (Steps A-D) is what makes single-tenant MicroPython runtime possible. Per-active-turn working set is now ~100 KiB; per-session baseline ~5 KiB; aggregate fits comfortably under an 8 MiB cgroup or a 512 MiB single-board Linux.

### Running on a device

On a real MicroPython target (ESP32-S3 8MB + WiFi is the reference), the standard `micropython` build is enough — you don't need the `coverage` variant. That's a contributor-only thing for running this repo's test gate.

```python
# 1. On your dev machine, install the modules exoclaw imports.
mpremote mip install asyncio dataclasses datetime typing __future__

# 2. Copy the package onto the device.
mpremote cp -r exoclaw :exoclaw/

# 3. Use it. Bring your own LLMProvider, Conversation, and channels —
#    those are protocols exoclaw doesn't ship implementations of.
mpremote run main.py
```

The unix port works the same way without `mpremote` — just put `exoclaw/` on `MICROPYPATH`.

What you don't have to think about:

- **Branching in your code.** Import from `exoclaw.executor` / `exoclaw.agent.loop` / etc. as normal; the shim handles the runtime split internally.
- **Streaming tools.** `async def execute_streaming(): yield ...` works on both runtimes. On MP the executor drains via plain `for` (uasyncio doesn't ship the async-iterator protocol) but the tool author writes the same code.
- **Per-task isolation.** `_compat.TaskLocal` keys storage by `id(asyncio.current_task())` on MP, so concurrent turns don't cross-wire each other's `turn.id` / `session.key` / scratch paths.

Plugins haven't been ported. The story today is: **core runs cleanly; deploy your own LLM provider + lightweight tool set**. A reference micro-friendly provider (`exoclaw-provider-micro-openai`, ~300 lines around `urequests`/`ussl`) is on the roadmap, and [`exoclaw-plugins`](https://github.com/Clause-Logic/exoclaw-plugins) tracks its own port separately.

### Contributing — running the test gate locally

The CI matrix needs `sys.settrace`, which the brew bottle of MicroPython doesn't enable. To run the gate locally, build the unix port's `coverage` variant from source:

```bash
git clone --depth 1 https://github.com/micropython/micropython ~/dev/micropython
cd ~/dev/micropython/ports/unix
# Freeze asyncio into the coverage build (the standard variant has it; coverage doesn't).
echo 'include("$(MPY_DIR)/extmod/asyncio")' >> variants/coverage/manifest.py
make submodules && make VARIANT=coverage -j$(nproc)

# Tell exoclaw where to find it (or set in mise.local.toml under [env]).
export EXOCLAW_MICROPYTHON_BIN=~/dev/micropython/ports/unix/build-coverage/micropython
```

Then from the exoclaw repo:

```bash
mise run test-micro    # full matrix entry — CPython-side wrapper drives the runner
mise run mp-shell      # drop into a REPL with exoclaw on MICROPYPATH
mise run mp-run        # run the runner directly, prints JSON report
```

CI does the same build + asyncio manifest patch once per PR (cached).

---

## License

MIT
