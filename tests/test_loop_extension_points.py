"""Tests for AgentLoop extension points: callbacks, ToolContext, execute_with_context, set_bus."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.queue import MessageBus


def _make_response(content="hello", has_tool_calls=False, finish_reason="stop"):
    r = MagicMock()
    r.has_tool_calls = has_tool_calls
    r.content = content
    r.finish_reason = finish_reason
    r.tool_calls = []
    r.reasoning_content = None
    r.thinking_blocks = None
    return r


def _make_tool_call(name="my_tool", args=None, call_id="tc1"):
    tc = MagicMock()
    tc.name = name
    tc.arguments = args or {}
    tc.id = call_id
    return tc


def _make_loop(**kwargs):
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(return_value=_make_response())
    conversation = MagicMock()
    conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    conversation.record = AsyncMock()
    conversation.clear = AsyncMock(return_value=True)
    loop = AgentLoop(bus=bus, provider=provider, conversation=conversation, **kwargs)
    return loop, bus


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------

class TestToolContext:
    def test_fields(self):
        ctx = ToolContext(session_key="cli:main", channel="cli", chat_id="main")
        assert ctx.session_key == "cli:main"
        assert ctx.channel == "cli"
        assert ctx.chat_id == "main"


# ---------------------------------------------------------------------------
# execute_with_context in ToolRegistry
# ---------------------------------------------------------------------------

class TestExecuteWithContext:
    async def test_calls_execute_with_context_when_present(self):
        reg = ToolRegistry()
        tool = MagicMock()
        tool.name = "ctx_tool"
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="plain")
        tool.execute_with_context = AsyncMock(return_value="context-aware result")
        reg.register(tool)

        ctx = ToolContext(session_key="s", channel="cli", chat_id="main")
        result = await reg.execute("ctx_tool", {}, ctx=ctx)
        assert result == "context-aware result"
        tool.execute_with_context.assert_called_once_with(ctx)
        tool.execute.assert_not_called()

    async def test_falls_back_to_execute_when_no_ctx(self):
        reg = ToolRegistry()
        tool = MagicMock()
        tool.name = "ctx_tool"
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="plain")
        tool.execute_with_context = AsyncMock(return_value="context-aware result")
        reg.register(tool)

        result = await reg.execute("ctx_tool", {}, ctx=None)
        assert result == "plain"
        tool.execute_with_context.assert_not_called()

    async def test_falls_back_when_tool_has_no_execute_with_context(self):
        reg = ToolRegistry()
        tool = MagicMock(spec=["name", "description", "parameters", "execute",
                                "cast_params", "validate_params"])
        tool.name = "simple_tool"
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="simple result")
        reg.register(tool)

        ctx = ToolContext(session_key="s", channel="cli", chat_id="main")
        result = await reg.execute("simple_tool", {}, ctx=ctx)
        assert result == "simple result"


# ---------------------------------------------------------------------------
# set_bus — duck-typed hook called at registration
# ---------------------------------------------------------------------------

class TestSetBus:
    def test_set_bus_called_on_tool_registration(self):
        tool = MagicMock()
        tool.name = "bus_tool"
        tool.set_bus = MagicMock()

        loop, bus = _make_loop(tools=[tool])
        tool.set_bus.assert_called_once_with(bus)

    def test_tools_without_set_bus_are_unaffected(self):
        tool = MagicMock(spec=["name", "description", "parameters", "execute"])
        tool.name = "plain_tool"
        # Should not raise even though set_bus is not implemented
        loop, _ = _make_loop(tools=[tool])


# ---------------------------------------------------------------------------
# Injectable registry
# ---------------------------------------------------------------------------

class TestInjectableRegistry:
    def test_injected_registry_is_used(self):
        reg = ToolRegistry()
        loop, _ = _make_loop(registry=reg)
        assert loop.tools is reg

    def test_default_registry_created_when_not_injected(self):
        loop, _ = _make_loop()
        assert isinstance(loop.tools, ToolRegistry)


# ---------------------------------------------------------------------------
# on_pre_context callback
# ---------------------------------------------------------------------------

class TestOnPreContext:
    async def test_on_pre_context_result_appended_to_plugin_context(self):
        extra_ctx = "## Extra\nsome injected context"
        on_pre_context = AsyncMock(return_value=extra_ctx)

        loop, bus = _make_loop(on_pre_context=on_pre_context)

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hello")
        await loop._process_message(msg)

        on_pre_context.assert_called_once()
        args = on_pre_context.call_args[0]
        assert args[0] == "hello"  # message
        assert args[2] == "cli"    # channel

    async def test_empty_pre_context_not_appended(self):
        on_pre_context = AsyncMock(return_value="")
        loop, bus = _make_loop(on_pre_context=on_pre_context)

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hello")
        await loop._process_message(msg)

        # build_prompt should be called — context just won't have extra
        loop.conversation.build_prompt.assert_called_once()


# ---------------------------------------------------------------------------
# on_pre_tool callback
# ---------------------------------------------------------------------------

class TestOnPreTool:
    async def test_on_pre_tool_rejection_used_as_result(self):
        tool_response = _make_response(has_tool_calls=True)
        tool_response.tool_calls = [_make_tool_call("my_tool")]
        final_response = _make_response(content="done")

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=[tool_response, final_response])

        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "A tool"
        tool.parameters = {"type": "object", "properties": {}}
        tool.execute = AsyncMock(return_value="real result")

        rejection_msg = "Rejected: not allowed"
        on_pre_tool = AsyncMock(return_value=rejection_msg)

        bus = MessageBus()
        loop = AgentLoop(
            bus=bus, provider=provider, conversation=conversation,
            tools=[tool], on_pre_tool=on_pre_tool,
        )

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="go")
        await loop._process_message(msg)

        on_pre_tool.assert_called_once_with("my_tool", {}, "cli:main")
        tool.execute.assert_not_called()

    async def test_on_pre_tool_pass_through_when_no_rejection(self):
        tool_response = _make_response(has_tool_calls=True)
        tool_response.tool_calls = [_make_tool_call("my_tool")]
        final_response = _make_response(content="done")

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=[tool_response, final_response])

        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        # spec excludes execute_with_context so registry falls back to execute()
        tool = MagicMock(spec=["name", "description", "parameters", "execute",
                                "cast_params", "validate_params"])
        tool.name = "my_tool"
        tool.description = "A tool"
        tool.parameters = {"type": "object", "properties": {}}
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="real result")

        on_pre_tool = AsyncMock(return_value=None)  # no rejection

        bus = MessageBus()
        loop = AgentLoop(
            bus=bus, provider=provider, conversation=conversation,
            tools=[tool], on_pre_tool=on_pre_tool,
        )

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="go")
        await loop._process_message(msg)

        tool.execute.assert_called_once()


# ---------------------------------------------------------------------------
# on_post_turn callback
# ---------------------------------------------------------------------------

class TestOnPostTurn:
    async def test_on_post_turn_fired_after_turn(self):
        on_post_turn = AsyncMock()
        loop, bus = _make_loop(on_post_turn=on_post_turn)

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hello")
        await loop._process_message(msg)

        # ensure_future schedules it — drain event loop
        await asyncio.sleep(0)
        on_post_turn.assert_called_once()
        args = on_post_turn.call_args[0]
        assert args[1] == "cli:main"   # session_key
        assert args[2] == "cli"        # channel
        assert args[3] == "main"       # chat_id


# ---------------------------------------------------------------------------
# on_max_iterations callback
# ---------------------------------------------------------------------------

class TestOnMaxIterations:
    async def test_on_max_iterations_fired_when_limit_reached(self):
        # Always return tool calls so the loop hits max_iterations
        tool_response = _make_response(has_tool_calls=True)
        tc = _make_tool_call("looping_tool")
        tool_response.tool_calls = [tc]

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(return_value=tool_response)

        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        tool = MagicMock()
        tool.name = "looping_tool"
        tool.description = "loops forever"
        tool.parameters = {"type": "object", "properties": {}}
        tool.execute = AsyncMock(return_value="still going")

        fired = asyncio.Event()

        async def _on_max(session_key: str, channel: str, chat_id: str) -> None:
            fired.set()

        bus = MessageBus()
        loop = AgentLoop(
            bus=bus, provider=provider, conversation=conversation,
            tools=[tool], max_iterations=2, on_max_iterations=_on_max,
        )

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="go")
        await loop._process_message(msg)
        await asyncio.sleep(0)

        assert fired.is_set(), "on_max_iterations was not called when iteration limit was reached"
