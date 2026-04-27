"""End-to-end agent-loop turn on MicroPython.

The agent loop's job: receive an InboundMessage, call the LLM,
optionally execute tool calls, record the turn, send an OutboundMessage.
These tests wire it up with a fake provider + Conversation that
satisfy the protocols without any network or storage — proving
exoclaw's *core* (the loop, bus, executor, registry) runs on MP.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

import asyncio

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import LLMResponse, ToolCallRequest


class _FakeProvider:
    """Single-shot LLM provider that returns a canned response.

    Exoclaw's ``LLMProvider`` protocol is structural — any class with
    these two async methods satisfies it."""

    def __init__(self, response):
        self._response = response

    def get_default_model(self):
        return "fake-model"

    async def chat(self, messages, tools=None, model=None, **kw):
        return self._response


class _ToolThenAnswerProvider:
    """Two-shot provider: first call asks for a tool, second returns
    the final answer. Exercises the loop's tool-dispatch iteration."""

    def __init__(self, tool_name, args):
        self._tool_name = tool_name
        self._args = args
        self._calls = 0

    def get_default_model(self):
        return "fake-model"

    async def chat(self, messages, tools=None, model=None, **kw):
        self._calls += 1
        if self._calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name=self._tool_name, arguments=self._args)
                ],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="all done", finish_reason="stop")


class _MemoryConversation:
    """Minimal Conversation impl: keeps the messages list in memory.

    Doesn't implement ``append`` so the loop uses the legacy
    end-of-turn ``record`` path."""

    def __init__(self):
        self._messages = []
        self.recorded = []

    async def build_prompt(self, session_id, message, **kw):
        # Return [...prior, user_message]. No prior on the first turn.
        return self._messages + [{"role": "user", "content": message}]

    async def record(self, session_id, new_messages):
        self.recorded.extend(new_messages)
        self._messages.extend(new_messages)

    async def clear(self, session_id):
        self._messages = []
        return True

    def list_sessions(self):
        return []


def test_agent_loop_constructs():
    """Plain construction — proves the imports + ``__init__`` succeed
    on MP. Catches issues like missing ``asyncio.Lock`` (which IS
    available on MP, but verify here as a smoke check)."""
    bus = MessageBus()
    provider = _FakeProvider(LLMResponse(content="x"))
    conv = _MemoryConversation()
    loop = AgentLoop(bus=bus, provider=provider, conversation=conv)
    assert loop.bus is bus
    assert loop.provider is provider
    assert loop.conversation is conv
    assert loop.model == "fake-model"
    assert loop.tools.tool_names == []


def test_agent_loop_processes_simple_turn():
    """Receive a message, call the (fake) LLM, return its content,
    record the turn. No tool calls, single iteration."""
    bus = MessageBus()
    provider = _FakeProvider(LLMResponse(content="hello back"))
    conv = _MemoryConversation()
    loop = AgentLoop(bus=bus, provider=provider, conversation=conv)

    async def _go():
        content, new_msgs = await loop.process_turn("session:x", "hello")
        assert content == "hello back", "expected 'hello back' got {!r}".format(content)
        # ``new_msgs`` should contain the user message + assistant reply.
        assert len(new_msgs) >= 2
        # Conversation got the new messages persisted via ``record``.
        assert len(conv.recorded) >= 2

    asyncio.run(_go())


def test_agent_loop_executes_tool_then_answers():
    """Provider asks for a tool first, gets back the result, then
    produces a final answer. Two iterations of the loop."""
    seen_args = []

    class _AddTool:
        name = "add"
        description = "adds two numbers"
        parameters = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }

        async def execute(self, **kw):
            seen_args.append(kw)
            return str(kw["a"] + kw["b"])

    bus = MessageBus()
    provider = _ToolThenAnswerProvider("add", {"a": 2, "b": 3})
    conv = _MemoryConversation()
    loop = AgentLoop(bus=bus, provider=provider, conversation=conv, tools=[_AddTool()])

    async def _go():
        content, new_msgs = await loop.process_turn("session:y", "add 2 + 3")
        assert content == "all done", "expected final answer 'all done' got {!r}".format(content)
        # The tool was called once with the LLM's args.
        assert seen_args == [{"a": 2, "b": 3}]
        # Two LLM calls (tool request, then final answer).
        assert provider._calls == 2

    asyncio.run(_go())
