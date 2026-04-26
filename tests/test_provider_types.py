"""Constructors for provider data types.

After converting ``ToolCallRequest`` / ``LLMResponse`` from
``@dataclass`` to plain classes (so they import on MicroPython,
which strips type annotations at compile time), the existing test
suite never exercises the constructors directly — most callers
build them inside provider plugins. These tests pin down the
constructor surface so the two runtimes stay aligned.
"""

from __future__ import annotations

from exoclaw.providers.types import LLMResponse, ToolCallRequest


def test_tool_call_request_constructor() -> None:
    req = ToolCallRequest(id="call-7", name="search", arguments={"q": "hi"})
    assert req.id == "call-7"
    assert req.name == "search"
    assert req.arguments == {"q": "hi"}


def test_llm_response_required_only() -> None:
    """Single required field; everything else gets a sensible default."""
    resp = LLMResponse(content="hello")
    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage == {}
    assert resp.reasoning_content is None
    assert resp.thinking_blocks is None
    assert resp.has_tool_calls is False


def test_llm_response_full() -> None:
    """All fields populated. ``has_tool_calls`` flips to True when any
    tool call is present."""
    tc = ToolCallRequest(id="x", name="t", arguments={})
    resp = LLMResponse(
        content=None,
        tool_calls=[tc],
        finish_reason="tool_calls",
        usage={"input_tokens": 100, "output_tokens": 50},
        reasoning_content="thinking...",
        thinking_blocks=[{"type": "thinking", "text": "..."}],
    )
    assert resp.content is None
    assert resp.tool_calls == [tc]
    assert resp.finish_reason == "tool_calls"
    assert resp.usage == {"input_tokens": 100, "output_tokens": 50}
    assert resp.reasoning_content == "thinking..."
    assert resp.thinking_blocks == [{"type": "thinking", "text": "..."}]
    assert resp.has_tool_calls is True
