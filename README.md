# exoclaw 🦀

**Protocol-only AI agent framework. Bring your own everything.**

exoclaw is the skeleton. You provide the flesh — your LLM provider, your conversation storage, your channels, your tools. Nothing is baked in.

```
pip install exoclaw
```

One runtime dependency: `loguru`.

---

## How it works

exoclaw is five protocols and a loop.

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

Tools are registered at construction time via `Nanobot(tools=[...])`. The loop calls `tool.execute(**args)` and feeds results back into the LLM context.

**Optional hooks** (duck-typed — implement if you need them):

```python
def on_inbound(self, msg: InboundMessage) -> None:
    """Called before each message is processed. Update per-turn state here."""

def system_context(self) -> str:
    """Return a string injected into the system prompt every turn."""

async def cancel_by_session(self, session_key: str) -> int:
    """Cancel running work for a session. Return count cancelled. Called on /stop."""

sent_in_turn: bool  # If True after execute(), loop suppresses the normal reply
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

## Usage

```python
import asyncio
from exoclaw import Nanobot

# All of these come from plugin packages — not from exoclaw itself
from exoclaw_provider_litellm import LiteLLMProvider
from exoclaw_conversation import DefaultConversation
from exoclaw_channel_telegram import TelegramChannel
from exoclaw_tools_web import WebSearchTool
from exoclaw_tools_mcp import connect_mcp

async def main():
    # MCP tools require async setup — connect before constructing the app
    mcp_tools = await connect_mcp({
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
    })

    app = Nanobot(
        provider=LiteLLMProvider(model="anthropic/claude-opus-4-5"),
        conversation=DefaultConversation(workspace="~/.mybot/workspace"),
        channels=[
            TelegramChannel(token="..."),
        ],
        tools=[
            WebSearchTool(),
            *mcp_tools,
        ],
    )
    await app.run()

asyncio.run(main())
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
  app.py                   # Nanobot — the composition root
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
