"""Tests for the skill-blind lifecycle-hook contract (exoclaw.agent.hooks)
and its wiring into AgentLoop.

Core is skill-blind: the loop only consults ``Conversation.active_hooks(event)``
— it never sees a "skill". These tests use a skill-blind stub conversation
that returns hooks conditionally, proving the loop fires them, applies
mutate/veto/inject, exposes ``run_context`` + ``run_effect``, and that an
absent/empty ``active_hooks`` fires nothing (so existing conversations are
unaffected). The actual skill→hook gating is tested in the conversation plugin.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.hooks import (
    BEFORE_FINISH,
    BEFORE_TOOL,
    BeforeFinishResult,
    BeforeToolResult,
    HookContext,
    HookRegistration,
    dispatch_before_finish,
    dispatch_before_tool,
    passthrough_effect,
)
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import LLMResponse, ToolCallRequest

Effect = Callable[..., Awaitable[object]]


async def _noop_effect(fn: Effect, *a: object, **kw: object) -> object:
    return await fn(*a, **kw)


def _ctx(event: str, **fields: object) -> HookContext:
    return HookContext(event=event, run_context={}, messages=[], run_effect=_noop_effect, **fields)


# ---------------------------------------------------------------------------
# Pure dispatcher semantics
# ---------------------------------------------------------------------------


def test_before_tool_mutations_compose_in_priority_order() -> None:
    """Higher priority runs first; each lower hook sees the args as mutated by
    the higher ones (true middleware composition)."""

    async def go() -> None:
        seen: list[tuple[str, dict[str, object]]] = []

        async def low(ctx: HookContext) -> BeforeToolResult:
            seen.append(("low", dict(ctx.params or {})))
            p = dict(ctx.params or {})
            p["low"] = 1
            return BeforeToolResult(params=p)

        async def high(ctx: HookContext) -> BeforeToolResult:
            seen.append(("high", dict(ctx.params or {})))
            p = dict(ctx.params or {})
            p["high"] = 1
            return BeforeToolResult(params=p)

        regs = [HookRegistration(low, priority=1), HookRegistration(high, priority=10)]
        res = await dispatch_before_tool(regs, _ctx(BEFORE_TOOL, params={"orig": 1}))

        assert [s[0] for s in seen] == ["high", "low"]  # priority order
        assert seen[1][1] == {"orig": 1, "high": 1}  # low saw high's mutation
        assert res.params == {"orig": 1, "high": 1, "low": 1}

    asyncio.run(go())


def test_before_tool_first_block_short_circuits() -> None:
    async def go() -> None:
        ran: list[str] = []

        async def blocker(ctx: HookContext) -> BeforeToolResult:
            return BeforeToolResult(block=True, block_reason="budget spent")

        async def after(ctx: HookContext) -> BeforeToolResult | None:
            ran.append("after")
            return None

        regs = [HookRegistration(blocker, priority=10), HookRegistration(after, priority=1)]
        res = await dispatch_before_tool(regs, _ctx(BEFORE_TOOL, params={}))

        assert res.block and res.block_reason == "budget spent"
        assert ran == []  # lower-priority hook never ran

    asyncio.run(go())


def test_passthrough_effect_runs_inline() -> None:
    """The default run_effect (for executors without one) just awaits the
    callable inline."""

    async def go() -> None:
        ran: list[int] = []

        async def eff(x: int) -> int:
            ran.append(x)
            return x * 2

        out = await passthrough_effect(eff, 21)
        assert out == 42
        assert ran == [21]

    asyncio.run(go())


def test_before_finish_highest_priority_nonempty_wins() -> None:
    async def go() -> None:
        async def low(ctx: HookContext) -> BeforeFinishResult:
            return BeforeFinishResult(continue_message="low")

        async def high(ctx: HookContext) -> BeforeFinishResult:
            return BeforeFinishResult(continue_message="high")

        regs = [HookRegistration(low, priority=1), HookRegistration(high, priority=10)]
        res = await dispatch_before_finish(regs, _ctx(BEFORE_FINISH))
        assert res.continue_message == "high"

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Loop integration via a skill-blind stub conversation
# ---------------------------------------------------------------------------


class _Provider:
    def __init__(self, replies: list[LLMResponse]) -> None:
        self._replies = list(replies)

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        **kw: object,
    ) -> LLMResponse:
        return self._replies.pop(0) if self._replies else LLMResponse(content="ok")


class _Conv:
    """Skill-blind: just returns whatever active_hooks/run_context it was
    handed. Stands in for a skill-aware conversation without core knowing."""

    def __init__(
        self,
        hooks: dict[str, list[HookRegistration]] | None = None,
        run_ctx: dict[str, object] | None = None,
    ) -> None:
        self._hooks = hooks or {}
        self._run_ctx = run_ctx or {}
        self.recorded: list[dict[str, object]] = []

    async def build_prompt(self, sid: str, message: str, **kw: object) -> list[dict[str, object]]:
        return [{"role": "user", "content": message}]

    async def record(self, sid: str, msgs: list[dict[str, object]]) -> None:
        self.recorded.extend(msgs)

    async def clear(self, sid: str) -> bool:
        return True

    def list_sessions(self) -> list[dict[str, object]]:
        return []

    def active_hooks(self, event: str) -> list[HookRegistration]:
        return self._hooks.get(event, [])

    def run_context(self) -> dict[str, object]:
        return self._run_ctx


class _RecordingTool:
    def __init__(self, name: str = "do") -> None:
        self.name = name
        self.description = "d"
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}
        self.received: dict[str, object] | None = None
        self.executed = False

    async def execute(self, **kwargs: object) -> str:
        self.executed = True
        self.received = kwargs
        return "tool-ok"


def _tool_call(name: str = "do", args: dict[str, object] | None = None) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id="tc1", name=name, arguments=args or {})],
        finish_reason="tool_calls",
    )


def test_loop_before_tool_hook_stamps_from_run_context() -> None:
    """A before_tool hook reads the authoritative cycle id from run_context and
    stamps it onto the tool args — the tool runs with the stamped args, not
    whatever the model passed. The cycle_id-stamping pattern, with zero loop
    knowledge of research."""

    async def go() -> None:
        tool = _RecordingTool()

        async def stamp(ctx: HookContext) -> BeforeToolResult:
            p = dict(ctx.params or {})
            p["cycle_id"] = ctx.run_context.get("cycle_id")
            return BeforeToolResult(params=p)

        conv = _Conv(
            hooks={BEFORE_TOOL: [HookRegistration(stamp)]},
            run_ctx={"cycle_id": "C1"},
        )
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=conv,
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed
        assert tool.received == {"q": "x", "cycle_id": "C1"}

    asyncio.run(go())


def test_loop_before_tool_hook_vetoes_call() -> None:
    """block=True refuses the tool; the model sees block_reason as the result
    and the tool never runs."""

    async def go() -> None:
        tool = _RecordingTool(name="web_search")

        async def veto(ctx: HookContext) -> BeforeToolResult:
            return BeforeToolResult(block=True, block_reason="budget spent — write up findings")

        conv = _Conv(hooks={BEFORE_TOOL: [HookRegistration(veto)]})
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(name="web_search"), LLMResponse(content="final")]),
            conversation=conv,
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed is False  # vetoed

    asyncio.run(go())


def test_loop_before_finish_hook_injects_and_continues() -> None:
    """A before_finish hook re-prompts a model that stopped without a required
    tool; the loop continues and ends on the next response."""

    async def go() -> None:
        seen: list[list[str]] = []

        async def nudge(ctx: HookContext) -> BeforeFinishResult:
            seen.append(list(ctx.tools_used or []))
            return BeforeFinishResult(
                continue_message="call finish first" if len(seen) == 1 else None
            )

        conv = _Conv(hooks={BEFORE_FINISH: [HookRegistration(nudge)]})
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([LLMResponse(content="partial"), LLMResponse(content="done")]),
            conversation=conv,
        )
        out = await loop.process_direct("go")
        assert out == "done"
        assert len(seen) == 2  # fired on both stops; continued after the first

    asyncio.run(go())


def test_loop_before_tool_hook_can_run_effect() -> None:
    """A hook can dispatch a side effect through HookContext.run_effect (the
    executor-backed seam durable executors journal). On DirectExecutor it runs
    inline."""

    async def go() -> None:
        ran: list[str] = []

        async def effectful(ctx: HookContext) -> BeforeToolResult | None:
            async def _record() -> None:
                ran.append("effect")

            await ctx.run_effect(_record)
            return None

        conv = _Conv(hooks={BEFORE_TOOL: [HookRegistration(effectful)]})
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(), LLMResponse(content="final")]),
            conversation=conv,
            tools=[_RecordingTool()],
        )
        await loop.process_direct("go")
        assert ran == ["effect"]

    asyncio.run(go())


def test_loop_fires_nothing_when_conversation_has_no_active_hooks() -> None:
    """A conversation without active_hooks (the common case) is unaffected — no
    hooks fire, the tool runs with the model's args unchanged."""

    async def go() -> None:
        class _Bare:
            async def build_prompt(
                self, sid: str, message: str, **kw: object
            ) -> list[dict[str, object]]:
                return [{"role": "user", "content": message}]

            async def record(self, sid: str, msgs: list[dict[str, object]]) -> None:
                pass

            async def clear(self, sid: str) -> bool:
                return True

            def list_sessions(self) -> list[dict[str, object]]:
                return []

        tool = _RecordingTool()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=_Bare(),
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed
        assert tool.received == {"q": "x"}  # unstamped — no hooks

    asyncio.run(go())


def test_conversation_protocol_hook_seams_default_empty() -> None:
    """The optional active_tools/active_hooks/run_context default to empty, so a
    conversation that doesn't override them contributes nothing."""

    class _Default(Conversation):
        async def build_prompt(
            self, session_id: str, message: str, **kw: object
        ) -> list[dict[str, object]]:
            return []

        async def record(self, session_id: str, new_messages: list[dict[str, object]]) -> None:
            return None

        async def clear(self, session_id: str) -> bool:
            return True

        def list_sessions(self) -> list[dict[str, object]]:
            return []

    c = _Default()
    assert c.active_tools() == set()
    assert c.active_hooks(BEFORE_TOOL) == []
    assert c.run_context() == {}


class _ThrowingHooks:
    """Stub whose hook seams raise — the loop must treat them as no-ops, never
    crash the turn (a buggy skill provider shouldn't take down a cycle)."""

    def __init__(self, raise_active: bool) -> None:
        self._raise_active = raise_active

    async def build_prompt(self, sid: str, message: str, **kw: object) -> list[dict[str, object]]:
        return [{"role": "user", "content": message}]

    async def record(self, sid: str, msgs: list[dict[str, object]]) -> None:
        pass

    async def clear(self, sid: str) -> bool:
        return True

    def list_sessions(self) -> list[dict[str, object]]:
        return []

    def active_hooks(self, event: str) -> list[HookRegistration]:
        if self._raise_active:
            raise RuntimeError("boom")

        async def noop(ctx: HookContext) -> None:
            return None

        return [HookRegistration(noop)]

    def run_context(self) -> dict[str, object]:
        raise RuntimeError("boom")


def test_loop_survives_throwing_active_hooks() -> None:
    async def go() -> None:
        tool = _RecordingTool()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=_ThrowingHooks(raise_active=True),
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed  # throwing active_hooks treated as no hooks

    asyncio.run(go())


def test_loop_survives_throwing_run_context() -> None:
    async def go() -> None:
        # active_hooks returns a benign hook, so _make_hook_context runs and
        # hits the throwing run_context — which must degrade to an empty bag.
        tool = _RecordingTool()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=_ThrowingHooks(raise_active=False),
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed

    asyncio.run(go())
