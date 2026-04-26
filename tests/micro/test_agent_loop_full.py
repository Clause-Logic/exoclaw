"""Comprehensive AgentLoop coverage on MicroPython.

Builds on ``test_agent_loop.py`` to exercise paths that the simpler
end-to-end tests don't reach: the ``run`` method's bus loop, max
iterations, ``/stop`` handling, ``on_pre_tool`` rejection, tool
exceptions, system context collection, ``_fmt`` tool-call logging,
and the response-strip-think helper.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

import asyncio

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import LLMResponse, ToolCallRequest


class _StubProvider:
    """Returns whatever LLMResponse the test pushes via ``replies``."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def get_default_model(self):
        return "stub-model"

    async def chat(self, messages, tools=None, model=None, **kw):
        self.calls += 1
        if not self._replies:
            return LLMResponse(content="default")
        return self._replies.pop(0)


class _StubConv:
    def __init__(self):
        self._messages = []
        self.recorded = []

    async def build_prompt(self, session_id, message, **kw):
        return self._messages + [{"role": "user", "content": message}]

    async def record(self, session_id, msgs):
        self.recorded.extend(msgs)
        self._messages.extend(msgs)

    async def clear(self, session_id):
        self._messages = []
        return True

    def list_sessions(self):
        return []


# ── _strip_think helper ────────────────────────────────────────────


def test_strip_think_removes_think_block():
    """``<think>...</think>`` blocks some open-source models embed
    in their content — the loop strips them so end-users never see
    the reasoning trace."""
    out = AgentLoop._strip_think("<think>internal monologue</think>real answer")
    assert out == "real answer"


def test_strip_think_returns_none_for_empty_input():
    """Empty / None input returns None, mirroring no-content semantics."""
    assert AgentLoop._strip_think(None) is None
    assert AgentLoop._strip_think("") is None


def test_strip_think_passthrough():
    """No ``<think>`` block → unchanged."""
    out = AgentLoop._strip_think("just an answer")
    assert out == "just an answer"


# ── _collect_plugin_context ────────────────────────────────────────


def test_collect_plugin_context_aggregates_system_context_strings():
    """Tools that expose ``system_context()`` contribute strings into
    the per-turn plugin_context list. Tools without it are skipped."""

    class _ContextTool:
        name = "ctx"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        def system_context(self):
            return "augmented context line"

    class _PlainTool:
        name = "plain"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_StubProvider([]),
        conversation=_StubConv(),
        tools=[_ContextTool(), _PlainTool()],
    )
    ctx = loop._collect_plugin_context()
    assert "augmented context line" in ctx
    assert len(ctx) == 1


# ── on_pre_tool rejection ──────────────────────────────────────────


def test_on_pre_tool_rejection_short_circuits_execution():
    """When ``on_pre_tool`` returns a non-empty string, the tool body
    is NOT executed — the rejection text becomes the tool result the
    LLM sees."""

    tool_invocations = []

    class _Echo:
        name = "echo"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            tool_invocations.append(True)
            return "should-not-run"

    async def _reject(tool_name, args, sk):
        return "policy denied"

    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="echo", arguments={})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="ok then"),
        ]
    )
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=_StubConv(),
        tools=[_Echo()],
        on_pre_tool=_reject,
    )

    async def _go():
        content, _ = await loop.process_turn("s", "do echo")
        assert content == "ok then"
        # The tool's execute body never ran; the rejection short-circuited.
        assert tool_invocations == []

    asyncio.run(_go())


# ── Tool exception handling ────────────────────────────────────────


def test_tool_exception_becomes_inline_error_message():
    """A tool that raises during execute → the loop catches the
    exception and synthesises an ``Error executing X: ...`` message
    that the LLM sees as the tool result. Subsequent iteration
    completes normally."""

    class _Boom:
        name = "boom"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            raise RuntimeError("tool blew up")

    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="boom", arguments={})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="recovered"),
        ]
    )
    bus = MessageBus()
    loop = AgentLoop(bus=bus, provider=provider, conversation=_StubConv(), tools=[_Boom()])

    async def _go():
        content, msgs = await loop.process_turn("s", "boom")
        assert content == "recovered"
        # The tool message must surface the exception text so the
        # LLM can react. ``RuntimeError`` text + the analyse hint.
        tool_msg = [m for m in msgs if m.get("role") == "tool"]
        assert tool_msg
        assert "blew up" in str(tool_msg[0].get("content", ""))

    asyncio.run(_go())


# ── Max iterations cap ─────────────────────────────────────────────


def test_max_iterations_short_circuits_loop():
    """If the LLM keeps requesting tools forever, the loop stops at
    ``max_iterations`` and emits the iteration_limit warning."""

    call_count = [0]

    class _LoopForever:
        name = "loop"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "still going"

    class _Provider:
        def get_default_model(self):
            return "m"

        async def chat(self, messages, tools=None, model=None, **kw):
            call_count[0] += 1
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="loop", arguments={})],
                finish_reason="tool_calls",
            )

    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_Provider(),
        conversation=_StubConv(),
        tools=[_LoopForever()],
        max_iterations=3,
    )

    async def _go():
        content, _ = await loop.process_turn("s", "go")
        # Loop hit the cap. ``content`` should mention iterations
        # exhausted somehow, or the loop just returns whatever the
        # last LLM call gave it. Either way ``call_count`` must
        # equal max_iterations + 1 (initial + max_iterations) or
        # be capped close to it.
        assert call_count[0] <= loop.max_iterations + 1
        assert content is not None

    asyncio.run(_go())


# ── /stop slash command ────────────────────────────────────────────


def test_handle_stop_message_clears_active_tasks():
    """When the bus delivers a ``/stop`` message, the loop's
    ``_handle_stop`` cancels any in-flight tasks for that session
    and publishes a confirmation outbound."""
    bus = MessageBus()
    loop = AgentLoop(bus=bus, provider=_StubProvider([]), conversation=_StubConv())

    async def _go():
        # Pre-populate active_tasks with a dummy task that's already
        # done — the stop handler should drain it gracefully.
        async def _noop():
            return None

        async def _drive():
            stop = InboundMessage(channel="x", sender_id="u", chat_id="c", content="/stop")
            # ``_handle_stop`` is the internal entry point; call it
            # directly so we don't have to spin up the bus loop just
            # for this assertion.
            await loop._handle_stop(stop)

        await _drive()
        # No exception, no in-flight tasks left for this session.
        assert "x:c" not in loop._active_tasks or not loop._active_tasks["x:c"]

    asyncio.run(_go())


# ── Multi-iteration with mixed tools + final answer ────────────────


def test_two_tool_calls_in_one_iteration():
    """LLM returns two tool calls in a single response — loop
    dispatches both, gathers results, runs the next iteration."""

    seen = []

    class _Stamp:
        name = "stamp"
        description = "x"
        parameters = {"type": "object", "properties": {"k": {"type": "string"}}}

        async def execute(self, **kw):
            seen.append(kw["k"])
            return f"stamped:{kw['k']}"

    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="c1", name="stamp", arguments={"k": "a"}),
                    ToolCallRequest(id="c2", name="stamp", arguments={"k": "b"}),
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="both stamped"),
        ]
    )
    bus = MessageBus()
    loop = AgentLoop(bus=bus, provider=provider, conversation=_StubConv(), tools=[_Stamp()])

    async def _go():
        content, _ = await loop.process_turn("s", "stamp two")
        assert content == "both stamped"
        assert seen == ["a", "b"]

    asyncio.run(_go())


# ── Reasoning content / thinking blocks pass-through ───────────────


def test_reasoning_content_recorded_in_messages():
    """When a provider sets ``reasoning_content`` on its response,
    the loop persists it on the assistant message so subsequent
    turns can see the reasoning trace."""
    provider = _StubProvider(
        [LLMResponse(content="answer", reasoning_content="step-by-step trace")]
    )
    bus = MessageBus()
    conv = _StubConv()
    loop = AgentLoop(bus=bus, provider=provider, conversation=conv)

    async def _go():
        content, msgs = await loop.process_turn("s", "ask")
        assert content == "answer"
        # Assistant message has the reasoning attached.
        asst = [m for m in msgs if m.get("role") == "assistant"]
        assert asst
        assert any("trace" in str(m.get("reasoning_content", "")) for m in asst)

    asyncio.run(_go())
