# exoclaw 🦀

**AI agent infrastructure that fits in your stack, not the other way around.**

You have an app. Wire in exoclaw and it becomes intelligent — tool use, session memory, multi-turn conversations, any LLM. You own every piece. Nothing is baked in.

```
pip install exoclaw
```

One runtime dependency: `loguru`.

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

exoclaw is six protocols and a loop.

```
InboundMessage → Bus → AgentLoop → LLM → Tools → Bus → OutboundMessage → Channel
```

1. A **Channel** receives a message from the outside world and puts it on the **Bus**
2. The **AgentLoop** pulls it off the bus, asks the **Conversation** to build a prompt
3. The prompt goes to the **LLMProvider**, which returns a response
4. If the response has tool calls, the loop executes them via registered **Tools**
5. The final response goes back on the bus, and the **Channel** delivers it

Every one of those nouns is a protocol. Swap any of them out. No inheritance required.

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

## License

MIT
