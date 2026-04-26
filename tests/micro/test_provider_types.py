"""Constructors for provider data types on MicroPython.

Mirrors ``tests/test_provider_types.py`` (the CPython side) so the
two runtimes hit the same constructor lines. ``ToolCallRequest`` /
``LLMResponse`` are plain classes (not ``@dataclass``) since
MicroPython strips type annotations at compile time.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

from exoclaw.providers.types import LLMResponse, ToolCallRequest


def test_tool_call_request_constructor():
    req = ToolCallRequest(id="call-7", name="search", arguments={"q": "hi"})
    assert req.id == "call-7"
    assert req.name == "search"
    assert req.arguments == {"q": "hi"}


def test_llm_response_required_only():
    resp = LLMResponse(content="hello")
    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage == {}
    assert resp.reasoning_content is None
    assert resp.thinking_blocks is None
    assert resp.has_tool_calls is False


def test_llm_response_with_tool_calls():
    tc = ToolCallRequest(id="x", name="t", arguments={})
    resp = LLMResponse(
        content=None,
        tool_calls=[tc],
        finish_reason="tool_calls",
        usage={"input_tokens": 100, "output_tokens": 50},
        reasoning_content="thinking...",
        thinking_blocks=[{"type": "thinking", "text": "..."}],
    )
    assert resp.has_tool_calls is True
    assert resp.usage == {"input_tokens": 100, "output_tokens": 50}
    assert resp.reasoning_content == "thinking..."
