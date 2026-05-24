"""Tests for the generic lifecycle-hook contract (exoclaw.agent.hooks) and its
wiring into AgentLoop.

Core asks the Conversation for one decision per seam (``before_tool`` /
``before_finish``) and applies it — it has no opinion on what produces that
decision or how multiple hooks compose into it (that lives in the consumer).
These tests use a stub conversation that returns a single decision directly,
proving the loop applies mutate/veto/inject, exposes ``run_context`` +
``run_effect``, and that an absent/raising decider is a no-op (so existing
conversations are unaffected and a buggy consumer can't take down a turn).
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.hooks import (
    BeforeFinishResult,
    BeforeToolResult,
    HookContext,
    passthrough_effect,
)
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import LLMResponse, ToolCallRequest

# A decider: takes the HookContext, returns a decision (or None).
BeforeTool = Callable[[HookContext], Awaitable["BeforeToolResult | None"]]
BeforeFinish = Callable[[HookContext], Awaitable["BeforeFinishResult | None"]]


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
    """Stub conversation that delegates the decider seams to the callables it
    was handed — stands in for a consumer that opts into hooks, without core
    knowing what's behind it (or how it composes multiple hooks)."""

    def __init__(
        self,
        before_tool: BeforeTool | None = None,
        before_finish: BeforeFinish | None = None,
        run_ctx: dict[str, object] | None = None,
    ) -> None:
        self._bt = before_tool
        self._bf = before_finish
        self._run_ctx = run_ctx or {}

    async def build_prompt(self, sid: str, message: str, **kw: object) -> list[dict[str, object]]:
        return [{"role": "user", "content": message}]

    async def record(self, sid: str, msgs: list[dict[str, object]]) -> None:
        pass

    async def clear(self, sid: str) -> bool:
        return True

    def list_sessions(self) -> list[dict[str, object]]:
        return []

    def run_context(self) -> dict[str, object]:
        return self._run_ctx

    async def before_tool(self, ctx: HookContext) -> BeforeToolResult | None:
        return await self._bt(ctx) if self._bt else None

    async def before_finish(self, ctx: HookContext) -> BeforeFinishResult | None:
        return await self._bf(ctx) if self._bf else None


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


def test_loop_before_tool_decider_stamps_from_run_context() -> None:
    """A before_tool decider reads the authoritative cycle id from run_context
    and stamps it onto the tool args — the tool runs with the stamped args, not
    whatever the model passed. The cycle_id-stamping pattern, with zero loop
    knowledge of the consumer."""

    async def go() -> None:
        tool = _RecordingTool()

        async def stamp(ctx: HookContext) -> BeforeToolResult:
            p = dict(ctx.params or {})
            p["cycle_id"] = ctx.run_context.get("cycle_id")
            return BeforeToolResult(params=p)

        conv = _Conv(before_tool=stamp, run_ctx={"cycle_id": "C1"})
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


def test_loop_before_tool_decider_vetoes_call() -> None:
    """block=True refuses the tool; the model sees block_reason as the result
    and the tool never runs."""

    async def go() -> None:
        tool = _RecordingTool(name="web_search")

        async def veto(ctx: HookContext) -> BeforeToolResult:
            return BeforeToolResult(block=True, block_reason="budget spent — write up findings")

        conv = _Conv(before_tool=veto)
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


def test_loop_before_finish_decider_injects_and_continues() -> None:
    """A before_finish decider re-prompts a model that stopped without a
    required tool; the loop continues and ends on the next response."""

    async def go() -> None:
        seen: list[list[str]] = []

        async def nudge(ctx: HookContext) -> BeforeFinishResult:
            seen.append(list(ctx.tools_used or []))
            return BeforeFinishResult(
                continue_message="call finish first" if len(seen) == 1 else None
            )

        conv = _Conv(before_finish=nudge)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([LLMResponse(content="partial"), LLMResponse(content="done")]),
            conversation=conv,
        )
        out = await loop.process_direct("go")
        assert out == "done"
        assert len(seen) == 2  # fired on both stops; continued after the first

    asyncio.run(go())


def test_loop_before_tool_decider_can_run_effect() -> None:
    """A decider can dispatch a side effect through HookContext.run_effect (the
    executor-backed seam durable executors journal). On DirectExecutor it runs
    inline."""

    async def go() -> None:
        ran: list[str] = []

        async def effectful(ctx: HookContext) -> BeforeToolResult | None:
            async def _record() -> None:
                ran.append("effect")

            await ctx.run_effect(_record)
            return None

        conv = _Conv(before_tool=effectful)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(), LLMResponse(content="final")]),
            conversation=conv,
            tools=[_RecordingTool()],
        )
        await loop.process_direct("go")
        assert ran == ["effect"]

    asyncio.run(go())


def test_loop_fires_nothing_when_conversation_has_no_deciders() -> None:
    """A conversation without before_tool/before_finish (the common case) is
    unaffected — no decision fires, the tool runs with the model's args."""

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
        assert tool.received == {"q": "x"}  # unstamped — no decider

    asyncio.run(go())


def test_conversation_protocol_decider_seams_default_none() -> None:
    """The optional decider seams (and active_tools/run_context) default to
    no-op, so a conversation that doesn't override them contributes nothing."""

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

    async def go() -> None:
        c = _Default()
        ctx = HookContext(
            event="before_tool", run_context={}, messages=[], run_effect=passthrough_effect
        )
        assert c.active_tools() == set()
        assert await c.before_tool(ctx) is None
        assert await c.before_finish(ctx) is None
        assert c.run_context() == {}

    asyncio.run(go())


class _ThrowingConv:
    """Decider seams (and run_context) raise — the loop must treat them as
    no-ops, never crash the turn (a buggy consumer shouldn't take down a turn)."""

    def __init__(self, raise_before_tool: bool) -> None:
        self._raise_before_tool = raise_before_tool

    async def build_prompt(self, sid: str, message: str, **kw: object) -> list[dict[str, object]]:
        return [{"role": "user", "content": message}]

    async def record(self, sid: str, msgs: list[dict[str, object]]) -> None:
        pass

    async def clear(self, sid: str) -> bool:
        return True

    def list_sessions(self) -> list[dict[str, object]]:
        return []

    async def before_tool(self, ctx: HookContext) -> BeforeToolResult | None:
        if self._raise_before_tool:
            raise RuntimeError("boom")
        return None  # benign — so run_context (which raises) still gets built

    def run_context(self) -> dict[str, object]:
        raise RuntimeError("boom")


def test_loop_survives_throwing_before_tool() -> None:
    async def go() -> None:
        tool = _RecordingTool()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=_ThrowingConv(raise_before_tool=True),
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed  # a throwing decider is treated as no decision

    asyncio.run(go())


def test_loop_survives_throwing_run_context() -> None:
    async def go() -> None:
        # before_tool returns None (benign), so the context gets built and hits
        # the throwing run_context — which must degrade to an empty bag.
        tool = _RecordingTool()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=_ThrowingConv(raise_before_tool=False),
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed

    asyncio.run(go())
