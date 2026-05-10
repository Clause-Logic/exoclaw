# exoclaw 🦀

[![PyPI](https://img.shields.io/pypi/v/exoclaw)](https://pypi.org/project/exoclaw/)
[![CI](https://github.com/Clause-Logic/exoclaw/actions/workflows/pr.yml/badge.svg)](https://github.com/Clause-Logic/exoclaw/actions/workflows/pr.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An AI agent that fits into your stack — not the other way around.**

You have an app. Wire in exoclaw and it gains tool use, conversation memory, and any LLM. You own every piece. Nothing baked in, nothing surprising.

```
pip install exoclaw
```

One runtime dependency: `structlog`. Around 2,000 lines of Python you can read in an afternoon.

---

## Want a working bot in 30 seconds?

```
pip install exoclaw-nanobot
exoclaw-nanobot
```

That bundles everything — provider, conversation, channels, tools — and gives you a working agent. The full plugin catalog lives at [exoclaw-plugins](https://github.com/Clause-Logic/exoclaw-plugins).

This repo is the protocol-only core that everything else builds on. Read on if you want to wire it into something you already have.

---

## Why exoclaw exists

exoclaw is a fork of [nanobot](https://github.com/NanobotAI/nanobot), stripped down to the agent loop and the protocols around it.

The original ships with batteries — provider, memory, cron, MCP, Telegram, Discord. Convenient to start. But every baked-in feature becomes a release-blocker: a Telegram API change holds up a cron bug fix, an MCP upgrade pulls in conflicts for users who don't even use MCP.

exoclaw cuts the knot. Seven protocols, one loop. Storage, channels, tools, providers — they all live in separate packages you opt into. The core never changes because it has nothing to change.

- **Auditable.** ~2,000 lines, mypy strict, 95% test coverage.
- **No dependency drag.** Your tree contains exactly what you chose.
- **No surprise breakage.** A bug in someone else's Telegram plugin can't break your app.
- **Composable.** Swap providers, storage, or channels without touching the loop.

---

## How it works

```
InboundMessage → Bus → AgentLoop → LLM → Tools → Bus → OutboundMessage → Channel
```

1. A **Channel** receives a message and puts it on the **Bus**.
2. The **AgentLoop** picks it up, asks the **Conversation** to build a prompt.
3. The prompt goes to the **LLMProvider**, which returns a response.
4. If the response calls **Tools**, the loop runs them and feeds the results back.
5. The final response goes back on the bus, and the **Channel** delivers it.

Everything underlined is a Python protocol. Pick the implementations you want from [exoclaw-plugins](https://github.com/Clause-Logic/exoclaw-plugins), or write your own.

---

## Use it

### Drop into your existing FastAPI app

exoclaw doesn't own your event loop — it runs as a background task while your routes act as producers and consumers on the bus.

```python
from fastapi import FastAPI
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.bus.events import InboundMessage, OutboundMessage

# These come from plugin packages — see exoclaw-plugins
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_conversation.conversation import DefaultConversation

app = FastAPI()
bus = MessageBus()
provider = LiteLLMProvider(default_model="claude-sonnet-4-6")
conversation = DefaultConversation.create(workspace="~/.mybot", provider=provider)
agent = AgentLoop(bus=bus, provider=provider, conversation=conversation)

@app.on_event("startup")
async def _start():
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

### Standalone Python script

```python
import asyncio
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.bus.events import InboundMessage
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_conversation.conversation import DefaultConversation

async def main():
    bus = MessageBus()
    provider = LiteLLMProvider(default_model="claude-sonnet-4-6")
    conversation = DefaultConversation.create(workspace="~/.mybot", provider=provider)
    loop = AgentLoop(bus=bus, provider=provider, conversation=conversation)
    asyncio.create_task(loop.run())

    await bus.publish_inbound(InboundMessage(
        channel="cli", sender_id="me", chat_id="main", content="Hello!",
    ))
    print((await bus.consume_outbound()).content)

asyncio.run(main())
```

### As a GitHub Actions bot

Zero infra. The bot replies to issues and PR comments using your `GITHUB_TOKEN` — no extra secrets needed. See [exoclaw-github](https://github.com/Clause-Logic/exoclaw-github) and the [live demo](https://github.com/Clause-Logic/exoclaw-github-demo).

```yaml
# .github/workflows/bot.yml
- uses: Clause-Logic/exoclaw-github@main
  with:
    trigger: "@exoclawbot"
    tools: github_pr_diff, github_file, github_checks, github_review, github_label
```

---

## Swap pieces without touching the rest

Every component sits behind a protocol. Change one without changing anything else:

```python
# File-backed sessions (default)
conversation = DefaultConversation.create(workspace="~/.mybot", ...)

# → swap for Redis without changing your AgentLoop, channels, or tools
from exoclaw_conversation_redis import RedisConversation
conversation = RedisConversation(url="redis://localhost", ...)
```

Same for providers (LiteLLM ↔ direct Anthropic ↔ local Ollama), the bus (asyncio queue ↔ Redis pub/sub), and the executor (inline ↔ Temporal ↔ Celery for durable execution and retries).

For durable execution under [Temporal](https://temporal.io) — every LLM call and tool execution checkpointed, survives worker death — see [exoclaw-temporal](https://github.com/Clause-Logic/exoclaw-temporal).

---

## Write your own tool

```python
class WeatherTool:
    name = "get_weather"
    description = "Get the current weather for a city."
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    async def execute(self, city: str) -> str:
        return f"It's sunny in {city}, 22°C."
```

Pass it to `AgentLoop(tools=[WeatherTool()])`. That's it. (Use `ToolBase` from `exoclaw.agent.tools.protocol` if you want parameter casting and JSON schema generation for free.)

Tools can also inject context into the system prompt each turn — useful for reminding the model about scheduled jobs, recent events, or anything else the agent should always know:

```python
class CronTool:
    name = "cron"
    # ... rest of the tool

    def system_context(self) -> str:
        return f"# Scheduled jobs\n\n{self._list_active_jobs()}"
```

---

## Write your own channel

```python
class WebhookChannel:
    name = "webhook"

    async def start(self, bus):
        self._bus = bus
        # start your web server, register routes, etc.

    async def stop(self): ...

    async def send(self, msg):
        # deliver msg.content to the webhook target
        ...

    async def _on_request(self, payload):
        await self._bus.publish_inbound(InboundMessage(
            channel=self.name,
            sender_id=payload["user_id"],
            chat_id=payload["chat_id"],
            content=payload["text"],
        ))
```

---

## MicroPython support

exoclaw core runs on **CPython and MicroPython** out of the same source tree. Every PR runs the test suite on both runtimes; both must hit ≥95% coverage on their own reachable lines. A change that breaks MicroPython compat fails CI even if CPython stays green.

In practice that means you can run a single-tenant agent on an ESP32-S3 (8MB) — per-active-turn working set is ~100 KiB, per-session baseline ~5 KiB.

```bash
# Install the modules exoclaw imports onto your device
mpremote mip install asyncio dataclasses datetime typing __future__

# Copy the package over and run it
mpremote cp -r exoclaw :exoclaw/
mpremote run main.py
```

Plugins haven't been ported yet — bring your own LLM provider and lightweight tools.

---

## License

MIT
