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

    async def test_model_override_reaches_provider(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))
        await loop._run_agent_loop([{"role": "user", "content": "hi"}], model="override-model")
        assert loop.provider.chat.call_args.kwargs["model"] == "override-model"

    async def test_no_override_falls_back_to_loop_default(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))
        await loop._run_agent_loop([{"role": "user", "content": "hi"}])
        assert loop.provider.chat.call_args.kwargs["model"] == loop.model

    async def test_process_direct_passes_model_through(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))
        await loop.process_direct("hi", model="override-model")
        assert loop.provider.chat.call_args.kwargs["model"] == "override-model"

    async def test_inbound_message_model_override_used(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="hi",
            model_override="from-channel",
        )
        await loop._process_message(msg)
        assert loop.provider.chat.call_args.kwargs["model"] == "from-channel"

    async def test_explicit_model_beats_msg_override(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="hi",
            model_override="from-channel",
        )
        await loop._process_message(msg, model="explicit")
        assert loop.provider.chat.call_args.kwargs["model"] == "explicit"

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

    async def test_preserves_pre_seeded_prior_source(self) -> None:
        """Phase 2b's lazy ``PriorSource`` must survive through
        ``_run_agent_loop``. Regression: an earlier version of
        this method unconditionally called
        ``self._executor.set_messages(initial_messages)`` at the
        top, overwriting the source ``build_prompt`` had just
        installed and defeating the phase 2b RAM reduction
        end-to-end. See ``docs/memory-model.md`` "Step A" and
        the integration test under
        ``exoclaw-nanobot/tests/test_phase_persistence_integration.py``.
        """
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))

        # Install a sentinel source, then run the loop.
        sentinel = lambda: [{"role": "user", "content": "from-source"}]  # noqa: E731
        loop._executor.set_prior_source(sentinel)

        await loop._run_agent_loop([{"role": "user", "content": "shouldnt-seed"}])

        # Source object identity must be unchanged — neither the
        # hot-path ``set_messages(initial_messages)`` nor any other
        # sneak-path replaced it.
        assert loop._executor._prior_var.get() is sentinel, (
            "_run_agent_loop replaced the PriorSource installed before "
            "the call — phase 2b auto-wire would never survive real "
            "production flow with this regression."
        )


# ---------------------------------------------------------------------------
# process_turn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    async def test_returns_content_and_new_messages(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response("done"))
        loop.conversation.build_prompt = AsyncMock(
            return_value=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        )

        final, new_msgs = await loop.process_turn("sess:1", "hi", channel="test", chat_id="c1")

        assert final == "done"
        # new_msgs should contain user message + assistant response
        assert len(new_msgs) >= 1
        assert any(m.get("role") == "assistant" for m in new_msgs)
        loop.conversation.record.assert_awaited_once()

    async def test_forwards_kwargs_to_build_prompt(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response("ok"))
        loop.conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])

        await loop.process_turn("s", "hi", channel="ch", chat_id="c", skills=["math"])

        _, call_kwargs = loop.conversation.build_prompt.call_args
        assert call_kwargs.get("skills") == ["math"]

    async def test_on_progress_forwarded(self) -> None:
        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response("ok"))
        loop.conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        progress = AsyncMock()

        await loop.process_turn("s", "hi", on_progress=progress)

        # No tool calls so progress shouldn't fire, but verify no error
        loop.conversation.record.assert_awaited_once()


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


# ---------------------------------------------------------------------------
# Per-message persistence path (Conversation.append)
# ---------------------------------------------------------------------------


def _tool_call_response(call_id: str = "tc1") -> MagicMock:
    """Assistant response that triggers one tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.name = "t1"
    tc.arguments = {"x": 1}
    r = _make_response(content="calling tool", has_tool_calls=True)
    r.tool_calls = [tc]
    return r


class TestPerMessagePersistence:
    """When the Conversation exposes ``append`` as a real coroutine, the
    loop flushes each assistant / tool / user message as it's produced
    — not batched at end-of-turn ``record``. Backwards compat: if no
    ``append`` is wired up, the legacy ``record(delta)`` path fires.

    This is the behaviour added for the 2026-04-23 OOM fix (exoclaw
    memory-model doc, phase 1): prevents the per-turn message buffer
    from being the sole holder of turn state for a whole turn.
    """

    def _make_append_loop(self) -> tuple[AgentLoop, MagicMock]:
        """AgentLoop with a Conversation whose append/post_turn/record
        are all ``AsyncMock`` so we can assert on them.

        AsyncMock is a coroutine function per
        ``asyncio.iscoroutinefunction``, so the loop's ``_has_append``
        check activates the per-message path — same as a real impl.
        """
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hello"}])
        conversation.append = AsyncMock()
        conversation.post_turn = AsyncMock()
        conversation.record = AsyncMock()
        conversation.clear = AsyncMock(return_value=True)
        loop = AgentLoop(bus=bus, provider=provider, conversation=conversation)
        # Register the tool the assistant will call. ``sent_in_turn`` is
        # explicitly False — leaving it as the MagicMock default truthy
        # value would trick process_direct into thinking the tool
        # already replied and swallow the final response.
        tool = MagicMock()
        tool.name = "t1"
        tool.sent_in_turn = False
        tool.get_definition = MagicMock(
            return_value={"type": "function", "function": {"name": "t1"}}
        )
        tool.execute_with_context = AsyncMock(return_value="tool output")
        tool.execute = AsyncMock(return_value="tool output")
        loop.tools._tools["t1"] = tool
        return loop, conversation

    async def test_append_called_for_user_and_assistant_and_tool(self) -> None:
        """One turn with one tool call should produce four ``append`` calls
        in order: user message, assistant-with-tool-call, tool result,
        final assistant response."""
        loop, conv = self._make_append_loop()
        # Two LLM responses: first triggers the tool, second is the final answer.
        loop.provider.chat = AsyncMock(
            side_effect=[_tool_call_response(), _make_response(content="done")]
        )

        result = await loop.process_direct("hello")
        assert result == "done"

        # Four appends in order.
        roles = [call.args[1]["role"] for call in conv.append.call_args_list]
        assert roles == ["user", "assistant", "tool", "assistant"], (
            f"expected user→assistant→tool→assistant, got {roles}"
        )
        # Session id threaded through to every append call.
        sids = {call.args[0] for call in conv.append.call_args_list}
        assert sids == {"cli:direct"}

    async def test_post_turn_fires_and_record_does_not(self) -> None:
        """When the append path is active, ``post_turn`` runs once at
        end-of-turn and the legacy ``record`` persistence is skipped
        entirely — we don't want to double-write the same messages."""
        loop, conv = self._make_append_loop()
        loop.provider.chat = AsyncMock(return_value=_make_response(content="ok"))

        await loop.process_direct("hi")

        conv.post_turn.assert_awaited_once_with("cli:direct")
        conv.record.assert_not_called()

    async def test_legacy_record_path_when_no_append(self) -> None:
        """Conversation implementations that predate ``append`` still work
        — the loop falls back to ``record(delta)`` at end-of-turn."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(return_value=_make_response(content="ok"))

        class _LegacyConversation:
            """No ``append`` or ``post_turn`` — legacy protocol only."""

            def __init__(self) -> None:
                self.recorded: list[list[dict[str, object]]] = []

            async def build_prompt(
                self, session_id: str, message: str, **kw: object
            ) -> list[dict[str, object]]:
                return [{"role": "user", "content": message}]

            async def record(self, session_id: str, new_messages: list[dict[str, object]]) -> None:
                self.recorded.append(new_messages)

            async def clear(self, session_id: str) -> bool:
                return True

            def list_sessions(self) -> list[dict[str, object]]:
                return []

        conv = _LegacyConversation()
        loop = AgentLoop(bus=bus, provider=provider, conversation=conv)

        await loop.process_direct("hi")

        assert len(conv.recorded) == 1, "legacy record must fire exactly once per turn"
        # Delta should include the user message + the assistant response.
        roles = [m["role"] for m in conv.recorded[0]]
        assert roles == ["user", "assistant"], f"unexpected delta shape: {roles}"
