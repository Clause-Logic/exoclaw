"""Tests for per-turn trace context binding.

Stage 1 of the turn-observability work: verifies that

1. ``DirectExecutor.mint_turn_id`` produces a valid uuidv7.
2. ``AgentLoop._process_turn_inline`` binds ``turn.id``,
   ``turn.root_id``, ``turn.parent_id``, ``turn.depth`` and
   ``turn.chain`` into structlog's contextvars for the duration of
   the turn and unbinds them on exit.
3. A nested ``_process_turn_inline`` call — the pattern used when a
   subagent re-enters the agent loop inside the same process —
   extends the ancestry instead of wiping it.

The whole point is: if these invariants hold, a single LogsQL query
``turn.root_id:<uuid>`` surfaces every log line emitted while handling
a user message, including every downstream subagent turn, without
joins or timestamp correlation.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock
from uuid import UUID

import structlog
import structlog.contextvars

from exoclaw.agent.loop import AgentLoop
from exoclaw.executor import DirectExecutor, _uuid7


class TestUuid7:
    def test_format_is_hyphenated_hex(self) -> None:
        u = _uuid7()
        assert len(u) == 36
        assert u.count("-") == 4
        parts = u.split("-")
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]

    def test_version_nibble_is_7(self) -> None:
        # Third group's first hex char is the version nibble.
        for _ in range(100):
            u = _uuid7()
            assert u.split("-")[2][0] == "7", u

    def test_variant_bits_are_10(self) -> None:
        # Fourth group's first hex char has variant in top 2 bits (binary 10xx
        # → hex 8/9/a/b).
        for _ in range(100):
            u = _uuid7()
            first_nibble = u.split("-")[3][0]
            assert first_nibble in "89ab", u

    def test_parses_as_uuid_and_reports_version_7(self) -> None:
        u = UUID(_uuid7())
        assert u.version == 7

    def test_is_time_ordered(self) -> None:
        # v7 embeds a ms timestamp in the high 48 bits — within a single
        # millisecond the random tail varies, but the timestamp prefix
        # must be monotonically non-decreasing across successive mints.
        # That's the weaker, actually-true RFC 9562 guarantee — per-id
        # strict monotonicity requires an optional counter extension we
        # don't implement.
        def _ts_prefix(u: str) -> str:
            # First 48 bits = first two hyphen-groups (8 + 4 hex chars).
            return u[:13]

        prefixes = [_ts_prefix(_uuid7()) for _ in range(50)]
        assert prefixes == sorted(prefixes), f"timestamp prefixes regressed: {prefixes}"

    def test_survives_wall_clock_regression(self) -> None:
        # NTP / VM-restore / leap seconds can push ``time.time_ns`` backwards.
        # The generator must clamp against a per-process high-water mark
        # so the id timestamp prefix still doesn't regress — otherwise
        # log-sort-by-turn.id breaks for any trace that straddles a
        # clock adjustment.
        import exoclaw.executor as ex_mod

        def _ts_prefix(u: str) -> str:
            return u[:13]

        before = _uuid7()

        real_time_ns = time.time_ns
        # Simulate wall clock jumping 10 seconds into the past.
        try:
            ex_mod.time.time_ns = lambda: real_time_ns() - 10_000_000_000  # type: ignore[attr-defined]
            after = _uuid7()
        finally:
            ex_mod.time.time_ns = real_time_ns

        assert _ts_prefix(before) <= _ts_prefix(after), (
            f"uuidv7 prefix regressed after simulated clock jump: before={before} after={after}"
        )


class TestDirectExecutorMintTurnId:
    async def test_returns_uuid7_string(self) -> None:
        executor = DirectExecutor()
        value = await executor.mint_turn_id()
        assert UUID(value).version == 7

    async def test_each_call_produces_unique_id(self) -> None:
        executor = DirectExecutor()
        ids = {await executor.mint_turn_id() for _ in range(100)}
        assert len(ids) == 100


def _make_loop(executor: DirectExecutor) -> AgentLoop:
    """Construct a minimal AgentLoop backed by mocks.

    ``_process_turn_inline`` calls ``build_prompt`` → ``_run_agent_loop``
    → ``record`` in that order. We only care about the contextvar
    bindings that happen around this pipeline, so everything is mocked
    to return minimal no-op values. The build_prompt fake returns a
    one-message list so ``new_msgs = all_msgs[len(initial) - 1 :]``
    slicing doesn't error.
    """
    conversation = MagicMock()
    loop = AgentLoop(
        bus=MagicMock(),
        provider=MagicMock(),
        conversation=conversation,
        executor=executor,
    )

    async def fake_build_prompt(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [{"role": "system", "content": "ctx"}]

    async def fake_run_loop(
        initial: list[dict[str, object]],
        on_progress: object = None,
        model: str | None = None,
        **_kwargs: object,
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        return ("response", [], list(initial))

    async def fake_record(*_args: object, **_kwargs: object) -> None:
        return None

    executor.build_prompt = fake_build_prompt  # type: ignore[method-assign]
    executor.record = fake_record  # type: ignore[method-assign]
    loop._run_agent_loop = fake_run_loop  # type: ignore[method-assign]
    return loop


class TestTurnContextBinding:
    async def test_binds_turn_context_during_turn(self) -> None:
        captured: dict[str, object] = {}

        async def _capture_inside_turn(
            _initial: list[dict[str, object]],
            on_progress: object = None,
            model: str | None = None,
            **_kwargs: object,
        ) -> tuple[str, list[str], list[dict[str, object]]]:
            captured.update(structlog.contextvars.get_contextvars())
            return ("response", [], [{"role": "system", "content": "ctx"}])

        executor = DirectExecutor()
        loop = _make_loop(executor)
        loop._run_agent_loop = _capture_inside_turn  # type: ignore[method-assign]

        structlog.contextvars.clear_contextvars()
        try:
            await loop._process_turn_inline("s:1", "hi")
        finally:
            structlog.contextvars.clear_contextvars()

        assert "turn.id" in captured
        turn_id = captured["turn.id"]
        assert isinstance(turn_id, str)
        assert UUID(turn_id).version == 7

        assert captured["turn.root_id"] == turn_id, "top-level turn must have root_id == turn_id"
        assert captured["turn.parent_id"] is None, "top-level turn has no parent"
        assert captured["turn.depth"] == 0
        assert captured["turn.chain"] == turn_id

    async def test_unbinds_turn_context_after_turn(self) -> None:
        executor = DirectExecutor()
        loop = _make_loop(executor)

        structlog.contextvars.clear_contextvars()
        try:
            await loop._process_turn_inline("s:1", "hi")
            after = structlog.contextvars.get_contextvars()
        finally:
            structlog.contextvars.clear_contextvars()

        for key in ("turn.id", "turn.root_id", "turn.parent_id", "turn.depth", "turn.chain"):
            assert key not in after, f"{key} leaked out of the turn"

    async def test_unbinds_turn_context_on_exception(self) -> None:
        executor = DirectExecutor()
        loop = _make_loop(executor)

        async def _raise(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

        loop._run_agent_loop = _raise  # type: ignore[method-assign]

        structlog.contextvars.clear_contextvars()
        try:
            try:
                await loop._process_turn_inline("s:1", "hi")
            except RuntimeError:
                pass
            after = structlog.contextvars.get_contextvars()
        finally:
            structlog.contextvars.clear_contextvars()

        for key in ("turn.id", "turn.root_id", "turn.parent_id", "turn.depth", "turn.chain"):
            assert key not in after, (
                f"{key} leaked out of the turn after exception — "
                "contextvars must be unbound in a finally block"
            )

    async def test_nested_turn_extends_ancestry(self) -> None:
        """A subagent re-entering ``_process_turn_inline`` must inherit
        the outer turn's root/chain rather than mint a fresh root.

        Simulates the stage-3 pattern where a child AgentLoop runs
        inside the same process: the parent's ``turn.*`` contextvars
        are still bound, the child calls ``_process_turn_inline``, and
        the child's binding must *extend* the chain instead of wiping
        it. We capture what the inner turn saw and assert on the
        ancestry.
        """
        executor = DirectExecutor()
        outer_loop = _make_loop(executor)
        inner_loop = _make_loop(executor)

        seen: dict[str, dict[str, object]] = {}

        async def outer_body(
            _initial: list[dict[str, object]],
            on_progress: object = None,
            model: str | None = None,
            **_kwargs: object,
        ) -> tuple[str, list[str], list[dict[str, object]]]:
            seen["outer"] = dict(structlog.contextvars.get_contextvars())
            # Re-enter the loop as if a subagent is handling a nested turn.
            await inner_loop._process_turn_inline("s:2", "nested")
            # Critical invariant: after the inner turn returns, the outer
            # turn's trace context must still be bound. Otherwise any
            # subsequent log line from the outer turn (including its own
            # ``turn_end``) would emit with no turn fields at all.
            seen["outer_after_inner"] = dict(structlog.contextvars.get_contextvars())
            return ("outer_resp", [], [{"role": "system", "content": "ctx"}])

        async def inner_body(
            _initial: list[dict[str, object]],
            on_progress: object = None,
            model: str | None = None,
            **_kwargs: object,
        ) -> tuple[str, list[str], list[dict[str, object]]]:
            seen["inner"] = dict(structlog.contextvars.get_contextvars())
            return ("inner_resp", [], [{"role": "system", "content": "ctx"}])

        outer_loop._run_agent_loop = outer_body  # type: ignore[method-assign]
        inner_loop._run_agent_loop = inner_body  # type: ignore[method-assign]

        structlog.contextvars.clear_contextvars()
        try:
            await outer_loop._process_turn_inline("s:1", "outer")
        finally:
            structlog.contextvars.clear_contextvars()

        outer = seen["outer"]
        inner = seen["inner"]

        assert outer["turn.root_id"] == outer["turn.id"]
        assert outer["turn.depth"] == 0
        assert outer["turn.parent_id"] is None
        assert outer["turn.chain"] == outer["turn.id"]

        assert inner["turn.root_id"] == outer["turn.root_id"], (
            "nested turn must inherit the outer's root_id"
        )
        assert inner["turn.parent_id"] == outer["turn.id"], (
            "nested turn's parent_id must point at the outer turn"
        )
        assert inner["turn.depth"] == 1
        assert inner["turn.chain"] == f"{outer['turn.chain']}:{inner['turn.id']}"
        assert inner["turn.id"] != outer["turn.id"]

        # After the inner turn returns, the outer turn's context must
        # be fully restored — every key back to the outer value, not
        # unbound, not left pointing at the inner turn.
        restored = seen["outer_after_inner"]
        for key in ("turn.id", "turn.root_id", "turn.parent_id", "turn.depth", "turn.chain"):
            assert restored.get(key) == outer.get(key), (
                f"{key} was not restored to the outer turn's value after "
                f"nested inner returned: got {restored.get(key)!r}, "
                f"expected {outer.get(key)!r}"
            )


class TestProcessMessagePreservesOuterTurnContext:
    """``_process_message`` must not wipe a pre-bound ``turn.*``
    context when it rebinds its session-level fields.

    This is the load-bearing invariant for stage-3 subagent trace
    propagation: ``SubagentManager._run`` binds ``turn.chain`` /
    ``turn.id`` / ``turn.root_id`` from the parent workflow
    arguments BEFORE calling ``loop.process_direct``. If
    ``_process_message`` internally called ``clear_contextvars()``,
    those bindings would be wiped before the child's
    ``_process_turn_inline`` read them — the child would see an
    empty context, mint a fresh root, and the subagent's log lines
    would drop out of the parent's ``turn.root_id`` query.

    Caught in production during the first deploy after stage 3
    landed: every ``depth=0`` turn in cli:direct sessions, zero
    ``depth>0`` turns observed.
    """

    async def test_outer_turn_chain_survives_process_message(self) -> None:
        from unittest.mock import AsyncMock

        from exoclaw.bus.events import InboundMessage

        executor = DirectExecutor()
        loop = _make_loop(executor)

        inside_turn: dict[str, object] = {}

        async def capture_inside_loop(
            _initial: list[dict[str, object]],
            on_progress: object = None,
            model: str | None = None,
            **_kwargs: object,
        ) -> tuple[str, list[str], list[dict[str, object]]]:
            inside_turn.update(structlog.contextvars.get_contextvars())
            return ("response", [], [{"role": "system", "content": "ctx"}])

        loop._run_agent_loop = capture_inside_loop  # type: ignore[method-assign]
        loop._collect_plugin_context = lambda: []  # type: ignore[method-assign,assignment]

        # Mock bus.publish_outbound to accept the OutboundMessage
        loop.bus.publish_outbound = AsyncMock()  # type: ignore[method-assign]

        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="hello",
        )

        # Simulate what SubagentManager._run does just before calling
        # loop.process_direct: seed the contextvars with parent ancestry
        # that should survive into the child's turn.
        structlog.contextvars.clear_contextvars()
        try:
            structlog.contextvars.bind_contextvars(
                **{
                    "turn.chain": "rootA:parentB",
                    "turn.id": "parentB",
                    "turn.root_id": "rootA",
                }
            )
            await loop._process_message(msg)
        finally:
            structlog.contextvars.clear_contextvars()

        # The inner (child) turn's ``_process_turn_inline`` should
        # have extended the outer chain, not started fresh. The seeded
        # parent chain ``rootA:parentB`` is already 2 segments deep
        # (root at depth 0, parentB at depth 1), so the new turn this
        # ``process_direct`` call creates is the grandchild at depth 2.
        assert inside_turn.get("turn.root_id") == "rootA", (
            f"child turn started a fresh root instead of inheriting "
            f"'rootA' — got {inside_turn.get('turn.root_id')!r}. "
            f"_process_message is wiping parent contextvars."
        )
        assert inside_turn.get("turn.parent_id") == "parentB"
        assert inside_turn.get("turn.depth") == 2
        chain = inside_turn.get("turn.chain", "")
        assert isinstance(chain, str)
        assert chain.startswith("rootA:parentB:"), (
            f"child chain must extend the outer, got {chain!r}"
        )

    async def test_system_message_preserves_outer_turn_chain(self) -> None:
        """Same invariant for the ``channel == "system"`` branch of
        ``_process_message`` — subagent result announcements from
        ``SubagentManager._announce_*`` flow through this path.
        """
        from unittest.mock import AsyncMock

        from exoclaw.bus.events import InboundMessage

        executor = DirectExecutor()
        loop = _make_loop(executor)

        inside_turn: dict[str, object] = {}

        async def capture_inside_loop(
            _initial: list[dict[str, object]],
            on_progress: object = None,
            model: str | None = None,
            **_kwargs: object,
        ) -> tuple[str, list[str], list[dict[str, object]]]:
            inside_turn.update(structlog.contextvars.get_contextvars())
            return ("response", [], [{"role": "system", "content": "ctx"}])

        loop._run_agent_loop = capture_inside_loop  # type: ignore[method-assign]
        loop._collect_plugin_context = lambda: []  # type: ignore[method-assign,assignment]
        loop.bus.publish_outbound = AsyncMock()  # type: ignore[method-assign]

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="cli:direct",
            content="[Subagent 'child' completed]",
        )

        structlog.contextvars.clear_contextvars()
        try:
            structlog.contextvars.bind_contextvars(
                **{
                    "turn.chain": "rootA:parentB",
                    "turn.id": "parentB",
                    "turn.root_id": "rootA",
                }
            )
            await loop._process_message(msg)
        finally:
            structlog.contextvars.clear_contextvars()

        assert inside_turn.get("turn.root_id") == "rootA"
        assert inside_turn.get("turn.parent_id") == "parentB"
        assert inside_turn.get("turn.depth") == 2
