"""Tests for exoclaw/agent/loop.py coverage."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.events import InboundMessage, OutboundMessage
from exoclaw.bus.queue import MessageBus


def _make_response(
    content: str | None = "hello",
    has_tool_calls: bool = False,
    finish_reason: str = "stop",
) -> MagicMock:
    r = MagicMock()
    r.has_tool_calls = has_tool_calls
    r.content = content
    r.finish_reason = finish_reason
    r.tool_calls = []
    r.reasoning_content = None
    r.thinking_blocks = None
    return r


def _make_loop(tools: list[object] | None = None) -> tuple[AgentLoop, MessageBus]:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(return_value=_make_response())
    conversation = MagicMock()
    conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    conversation.record = AsyncMock()
    conversation.clear = AsyncMock(return_value=True)
    loop = AgentLoop(bus=bus, provider=provider, conversation=conversation, tools=tools)
    return loop, bus


# ---------------------------------------------------------------------------
# _strip_think
# ---------------------------------------------------------------------------


class TestStripThink:
    def test_no_think_tags(self) -> None:
        assert AgentLoop._strip_think("hello world") == "hello world"

    def test_strips_think_block(self) -> None:
        result = AgentLoop._strip_think("<think>internal reasoning</think>answer")
        assert result == "answer"

    def test_strips_multiline_think_block(self) -> None:
        result = AgentLoop._strip_think("<think>\nline1\nline2\n</think>final")
        assert result == "final"

    def test_none_input(self) -> None:
        assert AgentLoop._strip_think(None) is None

    def test_empty_string(self) -> None:
        assert AgentLoop._strip_think("") is None

    def test_only_think_tags_returns_none(self) -> None:
        assert AgentLoop._strip_think("<think>only reasoning</think>") is None

    def test_whitespace_only_after_strip(self) -> None:
        assert AgentLoop._strip_think("<think>reasoning</think>   ") is None


# ---------------------------------------------------------------------------
# _tool_hint
# ---------------------------------------------------------------------------


class TestToolHint:
    def _tc(self, name: str, arguments: object) -> MagicMock:
        tc = MagicMock()
        tc.name = name
        tc.arguments = arguments
        return tc

    def test_single_tool_short_arg(self) -> None:
        tc = self._tc("web_search", {"query": "cats"})
        result = AgentLoop._tool_hint([tc])
        assert result == 'web_search("cats")'

    def test_single_tool_long_arg_truncated(self) -> None:
        long_val = "x" * 50
        tc = self._tc("web_search", {"query": long_val})
        result = AgentLoop._tool_hint([tc])
        assert '…"' in result
        assert "web_search" in result

    def test_multiple_tools(self) -> None:
        tc1 = self._tc("search", {"q": "cats"})
        tc2 = self._tc("fetch", {"url": "http://example.com"})
        result = AgentLoop._tool_hint([tc1, tc2])
        assert "search" in result
        assert "fetch" in result
        assert ", " in result

    def test_non_string_value_uses_name_only(self) -> None:
        tc = self._tc("my_tool", {"count": 42})
        result = AgentLoop._tool_hint([tc])
        assert result == "my_tool"

    def test_list_arguments(self) -> None:
        tc = self._tc("my_tool", [{"query": "hello"}])
        result = AgentLoop._tool_hint([tc])
        assert "my_tool" in result

    def test_empty_args(self) -> None:
        tc = self._tc("my_tool", {})
        result = AgentLoop._tool_hint([tc])
        assert result == "my_tool"

    def test_none_args(self) -> None:
        tc = self._tc("my_tool", None)
        result = AgentLoop._tool_hint([tc])
        assert result == "my_tool"


# ---------------------------------------------------------------------------
# _collect_plugin_context
# ---------------------------------------------------------------------------


class TestCollectPluginContext:
    def test_no_tools(self) -> None:
        loop, _ = _make_loop()
        assert loop._collect_plugin_context() == []

    def test_tool_with_system_context(self) -> None:
        loop, _ = _make_loop()
        tool = MagicMock()
        tool.name = "my_tool"
        tool.system_context = MagicMock(return_value="some context")
        loop.tools._tools["my_tool"] = tool

        result = loop._collect_plugin_context()
        assert result == ["some context"]

    def test_tool_without_system_context(self) -> None:
        loop, _ = _make_loop()
        tool = MagicMock(spec=["name", "execute"])
        tool.name = "plain_tool"
        loop.tools._tools["plain_tool"] = tool

        result = loop._collect_plugin_context()
        assert result == []

    def test_tool_system_context_returns_empty_string(self) -> None:
        loop, _ = _make_loop()
        tool = MagicMock()
        tool.name = "t"
        tool.system_context = MagicMock(return_value="")
        loop.tools._tools["t"] = tool

        result = loop._collect_plugin_context()
        assert result == []

    def test_tool_system_context_raises(self) -> None:
        loop, _ = _make_loop()
        tool = MagicMock()
        tool.name = "broken_tool"
        tool.system_context = MagicMock(side_effect=RuntimeError("boom"))
        loop.tools._tools["broken_tool"] = tool

        # Should not raise — errors are swallowed
        result = loop._collect_plugin_context()
        assert result == []

    def test_tool_system_context_returns_non_string(self) -> None:
        loop, _ = _make_loop()
        tool = MagicMock()
        tool.name = "t"
        tool.system_context = MagicMock(return_value=123)
        loop.tools._tools["t"] = tool

        result = loop._collect_plugin_context()
        assert result == []


# ---------------------------------------------------------------------------
# _run_agent_loop
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    async def test_normal_response_no_tool_calls(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="hello"))
        final, tools_used, msgs = await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assert final == "hello"
        assert tools_used == []

    async def test_error_finish_reason(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(
            return_value=_make_response(content="oops", finish_reason="error")
        )
        final, tools_used, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assert "error" in final.lower() or "oops" in final

    async def test_max_iterations_reached(self) -> None:
        loop, _ = _make_loop()
        loop.max_iterations = 2

        # Always return tool calls so it never stops
        tc = MagicMock()
        tc.id = "call1"
        tc.name = "dummy_tool"
        tc.arguments = {"x": "1"}
        resp = _make_response(has_tool_calls=True)
        resp.tool_calls = [tc]
        loop.provider.chat = AsyncMock(return_value=resp)
        loop.tools.execute = AsyncMock(return_value="result")

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assert "maximum" in final.lower()

    async def test_with_tool_calls(self) -> None:
        loop, _ = _make_loop()

        tc = MagicMock()
        tc.id = "call1"
        tc.name = "my_tool"
        tc.arguments = {"query": "test"}

        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]
        final_resp = _make_response(content="done")

        loop.provider.chat = AsyncMock(side_effect=[tool_resp, final_resp])
        loop.tools.execute = AsyncMock(return_value="tool result")

        final, tools_used, msgs = await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assert final == "done"
        assert "my_tool" in tools_used

    async def test_on_progress_called_for_tool_hint(self) -> None:
        loop, _ = _make_loop()

        tc = MagicMock()
        tc.id = "c1"
        tc.name = "search"
        tc.arguments = {"q": "test"}

        tool_resp = _make_response(content="thinking...", has_tool_calls=True)
        tool_resp.tool_calls = [tc]
        final_resp = _make_response(content="done")

        loop.provider.chat = AsyncMock(side_effect=[tool_resp, final_resp])
        loop.tools.execute = AsyncMock(return_value="result")

        progress_calls = []

        async def on_progress(content: str | None, **kw: object) -> None:
            progress_calls.append((content, kw))

        await loop._run_agent_loop([{"role": "user", "content": "hi"}], on_progress=on_progress)
        assert any(kw.get("tool_hint") for _, kw in progress_calls)

    async def test_reasoning_content_included(self) -> None:
        loop, _ = _make_loop()
        resp = _make_response(content="answer")
        resp.reasoning_content = "my reasoning"
        loop.provider.chat = AsyncMock(return_value=resp)

        final, _, msgs = await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assert final == "answer"
        # The assistant message should include reasoning_content
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assert any("reasoning_content" in m for m in assistant_msgs)

    async def test_thinking_blocks_included(self) -> None:
        loop, _ = _make_loop()
        resp = _make_response(content="answer")
        resp.thinking_blocks = [{"type": "thinking", "thinking": "..."}]
        loop.provider.chat = AsyncMock(return_value=resp)

        final, _, msgs = await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assert any("thinking_blocks" in m for m in assistant_msgs)

    async def test_strip_think_applied_to_content(self) -> None:
        loop, _ = _make_loop()
        resp = _make_response(content="<think>internal</think>actual answer")
        loop.provider.chat = AsyncMock(return_value=resp)

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assert final == "actual answer"
        assert "<think>" not in final


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    async def test_new_command_success(self) -> None:
        loop, _ = _make_loop()
        loop.conversation.clear = AsyncMock(return_value=True)
        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="/new")
        result = await loop._process_message(msg)
        assert result is not None
        assert "New session" in result.content

    async def test_new_command_failure(self) -> None:
        loop, _ = _make_loop()
        loop.conversation.clear = AsyncMock(return_value=False)
        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="/new")
        result = await loop._process_message(msg)
        assert "failed" in result.content.lower() or "Memory" in result.content

    async def test_help_command(self) -> None:
        loop, _ = _make_loop()
        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="/help")
        result = await loop._process_message(msg)
        assert result is not None
        assert "/new" in result.content

    async def test_regular_message(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="response text"))
        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="hello")
        result = await loop._process_message(msg)
        assert result is not None
        assert result.content == "response text"

    async def test_system_channel_message(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="done"))
        msg = InboundMessage(
            channel="system", sender_id="cron", chat_id="slack:c1", content="do thing"
        )
        result = await loop._process_message(msg)
        assert result is not None
        assert result.channel == "slack"
        assert result.chat_id == "c1"

    async def test_system_channel_no_colon_in_chat_id(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="done"))
        msg = InboundMessage(
            channel="system", sender_id="cron", chat_id="direct", content="do thing"
        )
        result = await loop._process_message(msg)
        assert result is not None
        assert result.channel == "cli"

    async def test_sent_in_turn_suppresses_reply(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="answer"))

        tool = MagicMock()
        tool.name = "notify_tool"
        tool.sent_in_turn = True
        loop.tools._tools["notify_tool"] = tool

        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="go")
        result = await loop._process_message(msg)
        assert result is None

    async def test_no_final_content_fallback(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(
            return_value=_make_response(content=None, finish_reason="stop")
        )
        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="hello")
        result = await loop._process_message(msg)
        assert result is not None
        assert result.content is not None

    async def test_notify_tools_inbound_called(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))

        tool = MagicMock()
        tool.name = "stateful_tool"
        tool.on_inbound = MagicMock()
        loop.tools._tools["stateful_tool"] = tool

        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="hi")
        await loop._process_message(msg)
        tool.on_inbound.assert_called_once_with(msg)

    async def test_bus_progress_used_when_on_progress_not_provided(self) -> None:
        """When called without on_progress (bus dispatch path), tool hints go to the bus."""
        loop, bus = _make_loop()

        r = _make_response(has_tool_calls=True)
        tc = MagicMock()
        tc.id = "t1"
        tc.name = "read_file"
        tc.arguments = {"path": "foo.txt"}
        r.tool_calls = [tc]
        loop.tools.execute = AsyncMock(return_value="file contents")

        final = _make_response(content="done")
        loop.provider.chat = AsyncMock(side_effect=[r, final])

        outbound: list[OutboundMessage] = []
        original_publish = bus.publish_outbound

        async def capture(msg: OutboundMessage) -> None:
            outbound.append(msg)
            await original_publish(msg)

        bus.publish_outbound = capture

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="go")
        # Called without on_progress — sentinel _UNSET → _bus_progress used
        await loop._process_message(msg)

        progress_msgs = [m for m in outbound if m.metadata and m.metadata.get("_progress")]
        assert progress_msgs, "Expected progress messages to be published to the bus"

    async def test_explicit_none_on_progress_suppresses_bus_progress(self) -> None:
        """When on_progress=None is passed explicitly (silent path), no progress hits the bus."""
        loop, bus = _make_loop()

        r = _make_response(has_tool_calls=True)
        tc = MagicMock()
        tc.id = "t1"
        tc.name = "read_file"
        tc.arguments = {"path": "foo.txt"}
        r.tool_calls = [tc]
        loop.tools.execute = AsyncMock(return_value="file contents")

        final = _make_response(content="done")
        loop.provider.chat = AsyncMock(side_effect=[r, final])

        outbound: list[OutboundMessage] = []
        original_publish = bus.publish_outbound

        async def capture(msg: OutboundMessage) -> None:
            outbound.append(msg)
            await original_publish(msg)

        bus.publish_outbound = capture

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="go")
        # Explicit None → silent, no _bus_progress
        await loop._process_message(msg, on_progress=None)

        progress_msgs = [m for m in outbound if m.metadata and m.metadata.get("_progress")]
        assert not progress_msgs, "Expected no progress messages when on_progress=None"


# ---------------------------------------------------------------------------
# process_direct
# ---------------------------------------------------------------------------


class TestProcessDirect:
    async def test_basic_call(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="direct response"))
        result = await loop.process_direct("hello")
        assert result == "direct response"

    async def test_returns_empty_string_on_none_response(self) -> None:
        loop, _ = _make_loop()
        # Suppress reply via sent_in_turn
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))
        tool = MagicMock()
        tool.name = "t"
        tool.sent_in_turn = True
        loop.tools._tools["t"] = tool
        result = await loop.process_direct("hello")
        assert result == ""

    async def test_custom_session_key(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))
        result = await loop.process_direct(
            "hello", session_key="custom:key", channel="api", chat_id="req1"
        )
        assert result == "ok"


# ---------------------------------------------------------------------------
# run / stop
# ---------------------------------------------------------------------------


class TestRunStop:
    async def test_stop_sets_running_false(self) -> None:
        loop, _ = _make_loop()
        assert not loop._running
        loop.stop()
        assert not loop._running

    async def test_run_starts_and_can_be_stopped(self) -> None:
        loop, bus = _make_loop()

        async def stop_soon() -> None:
            await asyncio.sleep(0.05)
            loop.stop()

        asyncio.create_task(stop_soon())
        # run() will loop until _running is False (set by stop_soon)
        await asyncio.wait_for(loop.run(), timeout=2.0)
        assert not loop._running

    async def test_run_dispatches_message(self) -> None:
        loop, bus = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="pong"))

        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="ping")

        async def feed_and_stop() -> None:
            await asyncio.sleep(0.01)
            await bus.publish_inbound(msg)
            await asyncio.sleep(0.1)
            loop.stop()

        asyncio.create_task(feed_and_stop())
        await asyncio.wait_for(loop.run(), timeout=2.0)

        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "pong"


# ---------------------------------------------------------------------------
# _dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    async def test_dispatch_exception_sends_error(self) -> None:
        loop, bus = _make_loop()
        loop._process_message = AsyncMock(side_effect=RuntimeError("boom"))
        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="hi")
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "error" in out.content.lower()

    async def test_dispatch_none_response_cli_publishes_empty(self) -> None:
        loop, bus = _make_loop()
        loop._process_message = AsyncMock(return_value=None)
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="hi")
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == ""

    async def test_dispatch_none_response_non_cli_no_publish(self) -> None:
        loop, bus = _make_loop()
        loop._process_message = AsyncMock(return_value=None)
        msg = InboundMessage(channel="slack", sender_id="u1", chat_id="c1", content="hi")
        await loop._dispatch(msg)
        # No message should be on the bus
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.1)
