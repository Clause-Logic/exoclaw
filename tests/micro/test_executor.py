"""DirectExecutor flow on MicroPython.

Proves the executor's per-turn message buffer (prior + delta), tool
dispatch, scratch-file lifecycle for streaming tools, and uuid7
generation all work end-to-end on MP.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

import asyncio

from exoclaw._compat import path_exists
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.executor import (
    DirectExecutor,
    ToolResult,
    _build_lazy_prior_source,
    _empty_prior_source,
    _supports_append,
    _supports_post_turn,
    _uuid7,
)


class _StaticTool:
    name = "static"
    description = "returns a fixed value"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kw):
        return "static-result"


def test_uuid7_format():
    """``_uuid7`` produces a UUID-v7 string: 8-4-4-4-12 hex with
    version nibble 7 in position 14 (the ``M`` of
    ``xxxxxxxx-xxxx-Mxxx-...``).  Same shape on both runtimes."""
    u = _uuid7()
    assert len(u) == 36, "expected 36-char UUID, got {} chars".format(len(u))
    assert u[8] == "-" and u[13] == "-" and u[18] == "-" and u[23] == "-"
    assert u[14] == "7", "version nibble should be 7, got {!r}".format(u[14])


def test_uuid7_monotonic():
    """Successive calls produce non-decreasing TIMESTAMP prefixes.

    The full ID isn't strictly sortable — the trailing 10 bytes are
    random per call, so two IDs in the same millisecond can have
    different lex order. The clock-regression guard inside
    ``_uuid7`` only constrains the timestamp prefix (first 13 hex
    chars: 12 timestamp + 1 version nibble), so that's what we
    compare."""
    a = _uuid7()
    b = _uuid7()
    # First 13 hex chars: ``xxxxxxxx-xxxx-7`` (8+4+1 minus the dashes).
    # The dashes are at positions 8 and 13, so first 8 hex + skip
    # dash + next 4 hex + skip dash + version nibble.
    a_ts = a[:8] + a[9:13] + a[14]
    b_ts = b[:8] + b[9:13] + b[14]
    assert a_ts <= b_ts, "timestamp prefix regressed: {} > {}".format(a_ts, b_ts)


def test_direct_executor_message_buffer_round_trip():
    """``set_messages`` seeds prior, ``append_messages`` adds delta,
    ``load_messages`` returns the concatenation in order. Mirrors the
    inline-execute path the agent loop uses every iteration."""
    ex = DirectExecutor()
    ex.set_messages([{"role": "system", "content": "you are helpful"}])
    ex.append_messages([{"role": "user", "content": "hi"}])
    ex.append_messages([{"role": "assistant", "content": "hey"}])
    msgs = ex.load_messages()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_direct_executor_set_messages_clears_delta():
    """``set_messages`` resets the prior + clears the delta. Sequential
    turns on the same executor + task share the TaskLocal binding —
    without the delta clear, turn 2 would see turn 1's delta leaked
    into ``load_messages``."""
    ex = DirectExecutor()
    ex.set_messages([{"role": "system", "content": "a"}])
    ex.append_messages([{"role": "user", "content": "u1"}])
    # Turn 2: re-seed prior. Delta from turn 1 should NOT appear.
    ex.set_messages([{"role": "system", "content": "b"}])
    msgs = ex.load_messages()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "b"


def test_execute_tool_inline_path():
    """The inline path for tools without ``execute_streaming``: the
    tool's ``execute`` is awaited and the result is returned as a
    plain ``ToolResult(content=..., content_file=None)``."""
    ex = DirectExecutor()
    reg = ToolRegistry()
    reg.register(_StaticTool())

    async def _go():
        result = await ex.execute_tool_with_handle(reg, "static", {})
        assert isinstance(result, ToolResult)
        assert result.content == "static-result"
        assert result.content_file is None

    asyncio.run(_go())


def test_execute_tool_unknown_returns_inline_error():
    """Unknown-tool error comes back as ``ToolResult(content=<msg>)``
    on the inline path — not raised."""
    ex = DirectExecutor()
    reg = ToolRegistry()

    async def _go():
        result = await ex.execute_tool_with_handle(reg, "missing", {})
        assert isinstance(result, ToolResult)
        assert "not found" in result.content.lower()
        assert result.content_file is None

    asyncio.run(_go())


def test_post_turn_runs_without_appendable_conversation():
    """``post_turn`` is a no-op when the Conversation doesn't implement
    ``post_turn`` (the legacy ``record``-only path). It should NOT
    raise — only Appendable Conversations get the hook fired."""
    ex = DirectExecutor()

    class _ConvWithoutPostTurn:
        async def record(self, session_id, msgs):
            pass

    async def _go():
        await ex.post_turn(_ConvWithoutPostTurn(), "session:x")

    asyncio.run(_go())


def test_empty_prior_source():
    """``_empty_prior_source`` is the seed used before any
    ``set_messages`` call — returns ``[]`` so a premature
    ``load_messages`` doesn't raise."""
    assert _empty_prior_source() == []


def test_build_lazy_prior_source_non_list_history():
    """A non-list ``history_snapshot`` (e.g. a Mock returning a
    truthy stand-in) → returns None so the executor falls back to
    the closure-over-list path."""
    out = _build_lazy_prior_source(
        full=[{"role": "user", "content": "hi"}],
        history_snapshot="not-a-list",  # Defensive type guard.
        reload_history=lambda: [],
    )
    assert out is None


def test_build_lazy_prior_source_empty_history():
    """Empty history snapshot → no lazy source needed."""
    out = _build_lazy_prior_source(
        full=[{"role": "user", "content": "hi"}],
        history_snapshot=[],
        reload_history=lambda: [],
    )
    assert out is None


def test_build_lazy_prior_source_history_too_large():
    """Snapshot longer than ``full`` can't fit anywhere in ``full``
    → returns None."""
    out = _build_lazy_prior_source(
        full=[{"role": "user", "content": "hi"}],
        history_snapshot=[{"a": 1}, {"b": 2}],
        reload_history=lambda: [],
    )
    assert out is None


def test_build_lazy_prior_source_finds_match_and_reloads():
    """Snapshot found inside ``full`` as a contiguous sublist →
    returns a callable that re-reads via ``reload_history`` on each
    invocation. Prefix and suffix close over the surrounding
    messages."""
    history = [{"role": "user", "content": "msg1"}, {"role": "assistant", "content": "r1"}]
    full = (
        [{"role": "system", "content": "sys"}] + history + [{"role": "user", "content": "current"}]
    )

    reload_count = [0]

    def _reload():
        reload_count[0] += 1
        return history

    src = _build_lazy_prior_source(full=full, history_snapshot=history, reload_history=_reload)
    assert src is not None
    out = src()
    # Reconstruct: prefix + reloaded + suffix.
    assert out == full
    assert reload_count[0] == 1
    # Second call re-reads — that's the point of the laziness.
    src()
    assert reload_count[0] == 2


def test_build_lazy_prior_source_no_match():
    """Snapshot doesn't appear as a contiguous sublist of full →
    None (caller falls back to closure-over-list)."""
    out = _build_lazy_prior_source(
        full=[{"role": "user", "content": "different"}],
        history_snapshot=[{"role": "user", "content": "missing"}],
        reload_history=lambda: [],
    )
    assert out is None


def test_supports_append_returns_false_for_non_async_method():
    """``_supports_append`` requires ``append`` to be a coroutine.
    Non-coroutine attribute (sync function) → False. Mirrors the
    defence against ``MagicMock`` auto-attrs in tests."""

    class _SyncAppend:
        def append(self, session_id, message):
            return None

    assert _supports_append(_SyncAppend()) is False


def test_supports_append_returns_true_for_async_method():
    """Coroutine ``append`` → True."""

    class _AsyncAppend:
        async def append(self, session_id, message):
            return None

    assert _supports_append(_AsyncAppend()) is True


def test_supports_post_turn_returns_false_for_missing_method():
    """No ``post_turn`` attr at all → False, no raise."""

    class _Empty:
        pass

    assert _supports_post_turn(_Empty()) is False


def test_post_turn_cleans_up_scratch_files():
    """Scratch files registered during the turn (via the streaming
    path) are unlinked at ``post_turn``. Simulates the cleanup path
    by directly populating the TaskLocal — the actual streaming
    path needs ``isasyncgenfunction`` which is stubbed False on MP."""
    from exoclaw._compat import make_scratch_path

    ex = DirectExecutor()
    # Hand-register scratch paths the same way the streaming path
    # would. This exercises the cleanup code without needing the
    # async-gen detection (which is conservative-False on MP).
    p1 = make_scratch_path(prefix="micro-scratch-", suffix=".txt")
    p2 = make_scratch_path(prefix="micro-scratch-", suffix=".txt")
    assert path_exists(p1)
    assert path_exists(p2)
    ex._scratch_paths_var.set([p1, p2])

    class _MinimalConv:
        async def record(self, session_id, msgs):
            pass

    async def _go():
        await ex.post_turn(_MinimalConv(), "s")

    asyncio.run(_go())
    assert not path_exists(p1), "scratch file p1 should have been removed"
    assert not path_exists(p2), "scratch file p2 should have been removed"
