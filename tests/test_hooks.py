"""Tests for the generic lifecycle-hook contract (exoclaw.agent.hooks) and its
wiring into AgentLoop.

Core asks the Conversation for one decision per seam (``before_tool`` /
``before_finish``) and applies it — it has no opinion on what produces that
decision or how multiple hooks compose into it (that lives in the consumer).
These tests use a stub conversation that returns a single decision directly,
proving the loop applies mutate/veto/inject, exposes ``run_context``, and that
an absent/raising decider is a no-op (so existing conversations are unaffected
and a buggy consumer can't take down a turn).
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.hooks import (
    BeforeFinishResult,
    BeforeToolResult,
    HookContext,
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


def test_loop_before_tool_inplace_param_mutation_does_not_change_call() -> None:
    """Mutating ``ctx.params`` in place must NOT change the tool call — only an
    explicit BeforeToolResult(params=...) does. The loop hands the decider a
    copy, so a hook that scribbles on ctx.params and returns None is a no-op."""

    async def go() -> None:
        tool = _RecordingTool()

        async def scribble(ctx: HookContext) -> None:
            (ctx.params or {})["evil"] = "injected"  # in-place, no result returned
            return None

        conv = _Conv(before_tool=scribble)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=conv,
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed
        assert tool.received == {"q": "x"}  # no "evil" key — the call is untouched

    asyncio.run(go())


def test_loop_hook_cannot_corrupt_run_context() -> None:
    """A decider mutating ``ctx.run_context`` must not touch the conversation's
    own bag — the loop hands the decider a shallow copy, not the live dict."""

    async def go() -> None:
        async def scribble(ctx: HookContext) -> None:
            ctx.run_context["evil"] = "injected"  # in-place
            return None

        conv = _Conv(before_tool=scribble, run_ctx={"cycle_id": "C1"})
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_Provider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=conv,
            tools=[_RecordingTool()],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert conv._run_ctx == {"cycle_id": "C1"}  # source bag untouched

    asyncio.run(go())


def test_loop_hook_cannot_corrupt_transcript_via_messages() -> None:
    """A decider that mutates ``ctx.messages`` must NOT change what the provider
    sees on later iterations — messages is a read-only per-dict copy of the
    transcript, not the executor's live buffer."""

    async def go() -> None:
        seen_roles: list[list[str]] = []

        class _RecProvider:
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
                seen_roles.append([str(m.get("role")) for m in messages])
                return self._replies.pop(0) if self._replies else LLMResponse(content="ok")

        async def corrupt(ctx: HookContext) -> None:
            for m in ctx.messages:
                m["role"] = "HACKED"
            return None

        conv = _Conv(before_tool=corrupt)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_RecProvider([_tool_call(args={"q": "x"}), LLMResponse(content="final")]),
            conversation=conv,
            tools=[_RecordingTool()],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        # The hook fired between iteration 1 and 2; iteration 2's prompt must not
        # carry the hook's scribble.
        assert all("HACKED" not in roles for roles in seen_roles)

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


def test_conversation_hook_seams_are_off_protocol_optin() -> None:
    """before_tool / before_finish / run_context are opt-in seams, NOT members
    of the Conversation Protocol. ``_Plain`` deliberately does NOT inherit
    Conversation — it's a structural impl mirroring exoclaw-turn's
    ``_EphemeralConversation`` (the real impl whose conformance 0.30.0 broke by
    putting the seams on the Protocol). It implements the required surface
    (incl. active_tools) and omits the seams; the ``Conversation`` annotation
    statically pins that this still conforms, and the loop reaches the seams via
    getattr with a no-op fallback so they stay absent here."""

    class _Plain:  # no Conversation base — exercises structural conformance
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

        def active_tools(self) -> set[str]:
            return set()

    c: Conversation = _Plain()  # ty statically verifies structural conformance
    assert isinstance(c, Conversation)  # runtime_checkable structural check
    # The hook seams are simply absent — opt-in only, reached via getattr.
    assert getattr(c, "before_tool", None) is None
    assert getattr(c, "before_finish", None) is None
    assert getattr(c, "run_context", None) is None


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
