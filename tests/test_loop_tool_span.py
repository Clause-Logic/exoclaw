"""Tests for the tool-call start/stop log pair in AgentLoop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import structlog

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.events import InboundMessage
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


def _make_tool_call(name: str = "my_tool", call_id: str = "tc1") -> MagicMock:
    tc = MagicMock()
    tc.name = name
    tc.arguments = {}
    tc.id = call_id
    return tc


def _make_loop(tool: object) -> AgentLoop:
    tool_response = _make_response(has_tool_calls=True)
    tool_response.tool_calls = [_make_tool_call("my_tool", call_id="tc-42")]
    final_response = _make_response(content="done")

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(side_effect=[tool_response, final_response])

    conversation = MagicMock()
    conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    conversation.record = AsyncMock()

    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        conversation=conversation,
        tools=[tool],
    )


def _tool(execute: AsyncMock) -> MagicMock:
    tool = MagicMock()
    tool.name = "my_tool"
    tool.description = "A tool"
    tool.parameters = {"type": "object", "properties": {}}
    tool.cast_params = MagicMock(side_effect=lambda p: p)
    tool.validate_params = MagicMock(return_value=[])
    tool.execute = execute
    # MagicMock auto-creates attrs; strip these so the registry doesn't
    # pick the execute_with_context branch (which would be non-awaitable).
    del tool.execute_with_context
    return tool


async def test_tool_result_logged_on_success() -> None:
    loop = _make_loop(_tool(AsyncMock(return_value="ok")))
    msg = InboundMessage(channel="cli", sender_id="u", chat_id="main", content="go")

    with structlog.testing.capture_logs() as logs:
        await loop._process_message(msg)

    events = {log["event"]: log for log in logs}
    assert "tool_call" in events
    assert "tool_result" in events

    start = events["tool_call"]
    stop = events["tool_result"]

    # Start/stop correlated by tool.call_id
    assert start["tool.call_id"] == "tc-42"
    assert stop["tool.call_id"] == "tc-42"
    assert start["tool.name"] == "my_tool"
    assert stop["tool.name"] == "my_tool"

    # Stop event carries action outcome
    assert stop["tool.status"] == "ok"
    assert "tool.duration_ms" in stop
    assert isinstance(stop["tool.duration_ms"], int)
    assert stop["log_level"] == "info"


async def test_tool_result_logged_on_error_captures_exception() -> None:
    loop = _make_loop(_tool(AsyncMock(side_effect=RuntimeError("kaboom"))))
    msg = InboundMessage(channel="cli", sender_id="u", chat_id="main", content="go")

    with structlog.testing.capture_logs() as logs:
        await loop._process_message(msg)

    stops = [log for log in logs if log["event"] == "tool_result"]
    assert len(stops) == 1
    stop = stops[0]

    assert stop["tool.status"] == "error"
    assert stop["tool.call_id"] == "tc-42"
    assert stop["log_level"] == "error"
    # exc_info must carry the actual exception instance (not True), since
    # the stop event is emitted from a finally block where sys.exc_info()
    # has already been cleared. Passing True would drop the traceback.
    exc_info = stop["exc_info"]
    assert isinstance(exc_info, RuntimeError)
    assert str(exc_info) == "kaboom"


async def test_tool_result_on_bare_exception_falls_back_to_type_name() -> None:
    # Exceptions with empty str(e) (e.g. no-arg TypeError) would otherwise
    # produce the useless "Error executing <tool>: " string this PR fixes.
    class _BareError(Exception):
        def __str__(self) -> str:
            return ""

    tool_response = _make_response(has_tool_calls=True)
    tool_response.tool_calls = [_make_tool_call("my_tool", call_id="tc-42")]
    final_response = _make_response(content="done")

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(side_effect=[tool_response, final_response])

    conversation = MagicMock()
    conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    conversation.record = AsyncMock()

    captured: list[str] = []

    async def on_tool_result(tc: object, result: str) -> None:
        captured.append(result)

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        conversation=conversation,
        tools=[_tool(AsyncMock(side_effect=_BareError()))],
        on_tool_result=on_tool_result,
    )

    msg = InboundMessage(channel="cli", sender_id="u", chat_id="main", content="go")
    await loop._process_message(msg)

    assert captured, "on_tool_result hook should have received the error string"
    assert "_BareError" in captured[0]


async def test_tool_exception_does_not_crash_loop() -> None:
    """Failing tool produces an LLM-visible error string, not a propagating exception."""
    loop = _make_loop(_tool(AsyncMock(side_effect=RuntimeError("kaboom"))))
    msg = InboundMessage(channel="cli", sender_id="u", chat_id="main", content="go")

    # Should not raise
    await loop._process_message(msg)


async def test_tool_reject_emits_rejected_status() -> None:
    tool_response = _make_response(has_tool_calls=True)
    tool_response.tool_calls = [_make_tool_call("my_tool", call_id="tc-99")]
    final_response = _make_response(content="done")

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(side_effect=[tool_response, final_response])

    conversation = MagicMock()
    conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    conversation.record = AsyncMock()

    tool = _tool(AsyncMock(return_value="ok"))
    on_pre_tool = AsyncMock(return_value="denied by policy")

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        conversation=conversation,
        tools=[tool],
        on_pre_tool=on_pre_tool,
    )

    msg = InboundMessage(channel="cli", sender_id="u", chat_id="main", content="go")
    with structlog.testing.capture_logs() as logs:
        await loop._process_message(msg)

    events = [log for log in logs if log["event"] in ("tool_reject", "tool_result")]
    by_event = {e["event"]: e for e in events}

    assert by_event["tool_reject"]["tool.call_id"] == "tc-99"
    assert by_event["tool_result"]["tool.status"] == "rejected"
    assert by_event["tool_result"]["tool.call_id"] == "tc-99"
    tool.execute.assert_not_called()
