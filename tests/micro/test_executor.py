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


def test_load_messages_before_set_messages_returns_empty():
    """First call to ``load_messages`` on a fresh executor (before
    ``set_messages``) hits the ``LookupError`` fallback inside
    ``_get_prior_source``: it installs ``_empty_prior_source`` and
    returns ``[]``. Same path the agent loop uses on a brand-new
    ``DirectExecutor`` instance."""
    ex = DirectExecutor()
    msgs = ex.load_messages()
    assert msgs == []


def test_append_messages_before_set_messages_initialises_delta():
    """``append_messages`` triggers the ``LookupError`` path inside
    ``_get_delta`` when called before any explicit set — the empty
    delta gets installed lazily."""
    ex = DirectExecutor()
    ex.append_messages([{"role": "user", "content": "hi"}])
    msgs = ex.load_messages()
    assert msgs == [{"role": "user", "content": "hi"}]


def test_set_prior_source_with_callable_replaces_prior():
    """``set_prior_source`` installs a custom callable that
    ``load_messages`` invokes on each call — the disk-backed-prior
    pattern (memory-model.md phase 2b)."""
    ex = DirectExecutor()
    call_count = [0]

    def _source():
        call_count[0] += 1
        return [{"role": "system", "content": "from-source"}]

    ex.set_prior_source(_source)
    ex.append_messages([{"role": "user", "content": "u"}])
    msgs = ex.load_messages()
    assert msgs[0]["content"] == "from-source"
    assert msgs[1]["content"] == "u"
    # Source was invoked once for this load_messages call.
    assert call_count[0] >= 1


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


def test_streaming_tool_drains_to_scratch_file_on_mp():
    """The memory-model.md Step D path on MicroPython: a tool with
    ``execute_streaming`` drains chunks to a per-turn scratch file
    rather than materialising the full content as one Python
    string. This is the prerequisite for the ESP32-S3 small-board
    target — without it, a single fat tool result can blow the box.

    On MP, ``execute_tool_with_handle`` uses **result-based dispatch**
    (calls the streamer and checks for ``__aiter__``) because
    ``inspect.isasyncgenfunction`` isn't available — the function
    object alone can't tell coroutines apart from async generators.
    """
    ex = DirectExecutor()
    reg = ToolRegistry()

    class _StreamingTool:
        name = "stream"
        description = "yields chunks"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            raise AssertionError("inline execute should be skipped")

        async def execute_streaming(self, **kw):
            # Big enough to overflow the 256-byte preview budget so
            # the truncation footer kicks in.
            yield "x" * 300
            yield "y" * 300

    reg.register(_StreamingTool())

    async def _go():
        outcome = await ex.execute_tool_with_handle(reg, "stream", {})
        # Streaming path: ``content_file`` is a real path, ``content``
        # is a preview ending in the size footer.
        assert outcome.content_file is not None
        assert path_exists(outcome.content_file)
        # Read the file and verify the full payload landed.
        with open(outcome.content_file) as fh:
            assert fh.read() == "x" * 300 + "y" * 300
        assert "xxx" in outcome.content
        assert "streamed 600 bytes" in outcome.content
        # Cleanup so the test doesn't leak the scratch file.
        import os as _os

        _os.remove(outcome.content_file)

    asyncio.run(_go())


def test_streaming_tool_call_id_sanitized_in_path():
    """``tool_call_id`` may contain unsafe path chars when it
    originates from an LLM/provider. The executor sanitises it
    before letting it near the filesystem (alnum + ``-`` / ``_``,
    capped at 64 chars). This test passes a tool_call_id with
    slashes / dots / null bytes — the resulting scratch-file path
    contains none of those."""
    ex = DirectExecutor()
    reg = ToolRegistry()

    class _S:
        name = "s"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        async def execute_streaming(self, **kw):
            yield "chunk"

    reg.register(_S())

    async def _go():
        outcome = await ex.execute_tool_with_handle(
            reg,
            "s",
            {},
            tool_call_id="../../etc/passwd\x00",
        )
        # Path got created, scrubbed of unsafe chars.
        assert outcome.content_file is not None
        assert "/etc/passwd" not in outcome.content_file
        assert "\x00" not in outcome.content_file
        # Cleanup.
        import os as _os

        try:
            _os.remove(outcome.content_file)
        except OSError:
            pass

    asyncio.run(_go())


def test_streaming_tool_non_string_chunks_coerced():
    """Tools that yield non-str chunks (numbers, dicts) get
    ``str(chunk)`` applied before the encode-and-write step. This
    keeps the executor robust to tools written without strict
    string-only output discipline."""
    ex = DirectExecutor()
    reg = ToolRegistry()

    class _S:
        name = "s"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        async def execute_streaming(self, **kw):
            yield 42
            yield {"a": 1}

    reg.register(_S())

    async def _go():
        outcome = await ex.execute_tool_with_handle(reg, "s", {})
        assert outcome.content_file is not None
        with open(outcome.content_file) as fh:
            written = fh.read()
        # Both chunks coerced to str and written.
        assert "42" in written
        assert "a" in written  # dict's str form contains the key
        import os as _os

        try:
            _os.remove(outcome.content_file)
        except OSError:
            pass

    asyncio.run(_go())


def test_streaming_tool_exception_unlinks_scratch_file():
    """An exception mid-stream → the executor unlinks the partial
    scratch file before re-raising. No half-written turds left
    behind for ``post_turn`` to find."""
    from exoclaw._compat import path_exists

    ex = DirectExecutor()
    reg = ToolRegistry()

    class _S:
        name = "boom"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        async def execute_streaming(self, **kw):
            yield "first chunk"
            raise RuntimeError("mid-stream failure")

    reg.register(_S())

    async def _go():
        try:
            await ex.execute_tool_with_handle(reg, "boom", {})
        except RuntimeError as e:
            assert "mid-stream" in str(e)
        else:
            raise AssertionError("expected RuntimeError to propagate")
        # No scratch file leaked. We can't easily get the path
        # back since the executor unlinked it, but we can verify
        # via the executor's tracked-paths var that nothing
        # accumulated (the cleanup ran before append).
        try:
            tracked = ex._scratch_paths_var.get()
        except LookupError:
            tracked = []
        assert all(not path_exists(p) for p in tracked)

    asyncio.run(_go())


def test_build_prompt_installs_lazy_prior_source_when_loader_is_sync():
    """When the Conversation exposes a sync ``load_persisted_history``,
    ``build_prompt`` auto-detects it and installs a disk-backed
    ``PriorSource`` — phase 2b of memory-model.md. Verifies the
    auto-wire fires and the lazy source replaces the closure-over-
    list snapshot."""
    ex = DirectExecutor()

    class _LazyConv:
        """Conversation with sync ``load_persisted_history`` — the
        signal for the executor to install a lazy prior source."""

        def __init__(self):
            self._prior = [
                {"role": "user", "content": "old1"},
                {"role": "assistant", "content": "old1-reply"},
            ]
            self.loader_calls = 0

        async def build_prompt(self, sid, message, **kw):
            return self._prior + [{"role": "user", "content": message}]

        def load_persisted_history(self, sid):
            self.loader_calls += 1
            return list(self._prior)

        async def record(self, sid, msgs):
            pass

        async def clear(self, sid):
            return True

        def list_sessions(self):
            return []

    conv = _LazyConv()

    async def _go():
        # First build_prompt should install a lazy source via
        # ``_build_lazy_prior_source``.
        result = await ex.build_prompt(conv, "s", "new")
        assert len(result) == 3
        # Loader was called by the lazy-source installer.
        assert conv.loader_calls >= 1
        # ``load_messages`` re-reads via the lazy source — calling
        # again increments the counter.
        before = conv.loader_calls
        ex.load_messages()
        assert conv.loader_calls > before

    asyncio.run(_go())


def test_streaming_tool_with_no_yield_does_not_crash():
    """If a tool defines ``execute_streaming`` but as a plain
    ``async def`` (no ``yield`` body), the executor doesn't crash.

    On CPython, ``inspect.isasyncgenfunction`` rejects it
    pre-call and the loop falls to the inline ``execute`` path —
    ``content == "fallthrough"``.

    On MicroPython, ``async def`` without ``yield`` compiles to a
    generator that ``for`` iterates as zero elements (MP's coroutine
    object has ``__next__`` and silently completes). The streaming
    path drains nothing and returns an empty preview with a
    scratch file. Neither runtime crashes — that's the contract."""
    ex = DirectExecutor()
    reg = ToolRegistry()

    class _MislabeledTool:
        name = "mislabel"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "fallthrough"

        async def execute_streaming(self, **kw):
            # Plain coroutine, NOT an async generator.
            return "coroutine-return"

    reg.register(_MislabeledTool())

    async def _go():
        outcome = await ex.execute_tool_with_handle(reg, "mislabel", {})
        # Doesn't crash. Either runtime path produces a ToolResult.
        assert isinstance(outcome.content, str)
        # Cleanup any scratch file produced on the MP path.
        if outcome.content_file is not None:
            import os as _os

            try:
                _os.remove(outcome.content_file)
            except OSError:
                pass

    asyncio.run(_go())


def test_post_turn_cleans_up_scratch_files():
    """Scratch files registered during a turn are unlinked at
    ``post_turn``. The set / get / unlink sequence happens inside
    the same task (matches production: streaming-tool dispatch
    populates the TaskLocal mid-turn, ``post_turn`` reads it at
    end-of-turn, both within ``process_turn``)."""
    from exoclaw._compat import make_scratch_path

    paths = []

    class _MinimalConv:
        async def record(self, session_id, msgs):
            pass

    async def _go():
        ex = DirectExecutor()
        # Register inside the task so the per-task TaskLocal storage
        # is the one ``post_turn`` will read from. CPython's
        # ContextVar inherits the outer context on ``asyncio.run``;
        # MP's per-task storage is keyed by ``id(current_task())``
        # so set + get must happen in the same task.
        p1 = make_scratch_path(prefix="micro-scratch-", suffix=".txt")
        p2 = make_scratch_path(prefix="micro-scratch-", suffix=".txt")
        paths.append(p1)
        paths.append(p2)
        assert path_exists(p1)
        assert path_exists(p2)
        ex._scratch_paths_var.set([p1, p2])
        await ex.post_turn(_MinimalConv(), "s")

    asyncio.run(_go())
    assert not path_exists(paths[0]), "scratch file p1 should have been removed"
    assert not path_exists(paths[1]), "scratch file p2 should have been removed"
