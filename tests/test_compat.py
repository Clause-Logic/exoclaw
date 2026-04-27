"""CPython side of the compat-shim test matrix.

Mirrors ``tests/micro/test_compat.py`` — same surface, opposite
runtime branches. The matrix layout:

- This file → CPython runtime, runs under pytest.
- ``tests/micro/test_compat.py`` → MicroPython runtime, runs under
  the subprocess driver in ``tests/_micropython_runner/run.py``,
  which is itself invoked by ``tests/test_micropython_runner.py``.

Each runtime independently asserts ≥95% coverage of the
``exoclaw/_compat.py`` lines reachable on that runtime. Per-runtime
exclusion is driven by the ``# pragma: no cover (cpython|micropython)``
markers in the source. See ``pyproject.toml``'s ``[tool.coverage.report]``
for the CPython side and ``test_micropython_runner.py``'s
``_executable_lines_for`` for the MicroPython side.
"""

from __future__ import annotations

import contextvars
import inspect
import io
import json
import os
import pathlib
import secrets
import sys
import tempfile
import threading
from collections.abc import AsyncIterator

import pytest

from exoclaw import _compat as c


def test_branch_selection_cpython() -> None:
    """Prove the CPython branches in ``_compat.py`` were taken at
    import time. Each helper resolves to the CPython stdlib
    implementation, NOT the micro fallback."""
    assert c.IS_MICROPYTHON is False
    assert c.random_bytes is secrets.token_bytes
    # ``make_lock()`` returns a real ``threading.Lock`` (not the
    # ``_NoopLock`` shim). Compare class because each ``Lock()``
    # call returns a distinct instance.
    assert c.make_lock().__class__ is threading.Lock().__class__
    assert c.TaskLocal is contextvars.ContextVar
    assert c.iscoroutinefunction is inspect.iscoroutinefunction
    assert c.isasyncgenfunction is inspect.isasyncgenfunction


def test_random_bytes_returns_correct_length() -> None:
    b = c.random_bytes(16)
    assert len(b) == 16
    assert isinstance(b, (bytes, bytearray))


def test_make_lock_context_manager() -> None:
    lock = c.make_lock()
    with lock:
        pass


def test_task_local_get_set_reset() -> None:
    var = c.TaskLocal("cpython_test")
    token = var.set(99)
    assert var.get() == 99
    var.reset(token)


def test_iscoroutinefunction_distinguishes_async_def() -> None:
    async def _async_fn() -> None:
        return None

    def _sync_fn() -> None:
        return None

    assert c.iscoroutinefunction(_async_fn) is True
    assert c.iscoroutinefunction(_sync_fn) is False


def test_isasyncgenfunction_distinguishes_async_gen() -> None:
    async def _async_gen() -> AsyncIterator[int]:
        yield 1

    async def _async_fn() -> None:
        return None

    assert c.isasyncgenfunction(_async_gen) is True
    assert c.isasyncgenfunction(_async_fn) is False


def test_make_scratch_path_default_dir() -> None:
    """No ``dir`` arg → resolved via ``tempfile.gettempdir`` on
    CPython. Covers the CPython-only branch in ``make_scratch_path``."""
    p = c.make_scratch_path(prefix="cpy-test-", suffix=".txt")
    try:
        assert os.path.exists(p)
        assert os.path.getsize(p) == 0
        assert os.path.dirname(p) == tempfile.gettempdir()
    finally:
        os.remove(p)


def test_make_scratch_path_explicit_dir() -> None:
    """Explicit ``dir`` arg → skips ``gettempdir``, uses caller path."""
    custom_dir = tempfile.mkdtemp()
    try:
        p = c.make_scratch_path(prefix="cpy-dir-", dir=custom_dir)
        assert os.path.dirname(p) == custom_dir
        os.remove(p)
    finally:
        os.rmdir(custom_dir)


def test_get_logger_returns_real_structlog() -> None:
    """On CPython with ``structlog`` installed (which it always is —
    it's a dependency), ``get_logger`` returns a real structlog
    logger. The stub fallback is reserved for micro / missing-dep
    environments and lives behind ``# pragma: no cover (cpython)``."""
    log = c.get_logger("test")
    assert hasattr(log, "bind")
    log.bind(extra="ok").info("smoke_test_cpython")


def test_micro_only_classes_callable_on_cpython() -> None:
    """``_NoopLock`` and ``_StubLogger`` are the MicroPython fallback
    classes. They're defined unconditionally at module top-level (so
    the file imports cleanly on both runtimes) but only INVOKED on
    micro. We construct + drive them directly here so the methods
    are exercised on CPython too — both for coverage and to catch
    regressions even when the MicroPython matrix entry isn't
    available locally."""
    # ── _NoopLock ────────────────────────────────────────────────
    lock = c._NoopLock()
    with lock:
        pass
    assert lock.acquire() is True
    assert lock.acquire(blocking=False) is True  # *args / **kwargs accepted
    assert lock.release() is None

    # ── _StubLogger ──────────────────────────────────────────────
    captured = io.StringIO()
    saved_stdout = sys.stdout
    sys.stdout = captured
    try:
        log = c._StubLogger("unit-test")
        log.debug("dbg_event", k=1)
        log.info("info_event", k=2)
        log.warning("warn_event")
        log.error("err_event", reason="x")
        log.exception("exc_event", exc="boom")
        # ``bind`` is a no-op that returns self.
        assert log.bind(extra="ok") is log
    finally:
        sys.stdout = saved_stdout

    lines = [ln for ln in captured.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 5
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0] == {"level": "debug", "event": "dbg_event", "logger": "unit-test", "k": 1}
    assert parsed[1]["level"] == "info"
    assert parsed[2]["level"] == "warning"
    assert parsed[3]["level"] == "error"
    # ``exception`` maps to error level (structlog convention is the
    # same — the level field stays ``error``, not ``exception``).
    assert parsed[4]["level"] == "error"
    assert parsed[4]["exc"] == "boom"
    # ``_emit`` falls back to a bare-print path when ``json.dumps``
    # raises. Force that branch by handing the logger an unencodable
    # field. ``object()`` is not JSON-serializable; the fallback
    # path catches the exception and prints ``[level] event``.
    captured2 = io.StringIO()
    sys.stdout = captured2
    try:
        c._StubLogger("fallback").info("err", bad=object())
    finally:
        sys.stdout = saved_stdout
    fallback_out = captured2.getvalue().strip()
    assert fallback_out.startswith("[info] err")


def test_make_scratch_path_handles_existing_dir() -> None:
    """``os.mkdir`` raises ``OSError`` when the target dir exists.
    The shim swallows that — covers the ``except OSError`` branch."""
    # Default ``dir=None`` → ``tempfile.gettempdir()`` which always
    # exists, so the ``mkdir`` call inside the shim hits the
    # ``except OSError`` path. Just verify the call succeeds.
    p = c.make_scratch_path(prefix="cpy-existing-")
    try:
        assert os.path.exists(p)
    finally:
        os.remove(p)


def test_path_helpers() -> None:
    """``path_basename`` returns the trailing component;
    ``path_exists`` mirrors ``os.path.exists`` via ``os.stat`` +
    ``OSError`` catch."""
    assert c.path_basename("/tmp/foo.txt") == "foo.txt"
    assert c.path_basename("foo.txt") == "foo.txt"  # no separator
    assert c.path_basename("/") == ""

    p = c.make_scratch_path(prefix="cpy-exists-")
    try:
        assert c.path_exists(p) is True
        os.remove(p)
        assert c.path_exists(p) is False
    finally:
        # Already removed inside the test body, but defensive — the
        # cleanup path can re-fail silently.
        try:
            os.remove(p)
        except OSError:
            pass


def test_monotonic_helpers() -> None:
    """``monotonic_ms`` returns int milliseconds; ``monotonic_diff_ms``
    is just subtraction on CPython (the wrap-safe path is MP-only)."""
    a = c.monotonic_ms()
    b = c.monotonic_ms()
    assert isinstance(a, int)
    assert isinstance(b, int)
    assert b >= a
    assert c.monotonic_diff_ms(b, a) == b - a


def test_async_queue_blocks_on_empty_get() -> None:
    """The ``_AsyncQueue.get`` loop awaits ``_event.wait`` when the
    list is empty. Verify the wait/clear cycle works by putting
    after a short delay and checking get unblocks."""
    import asyncio

    q = c._AsyncQueue()

    async def _go() -> None:
        async def _put_later() -> None:
            await asyncio.sleep(0.01)
            await q.put("hello")

        producer = asyncio.create_task(_put_later())
        # ``get`` will await the event since the queue is empty;
        # the producer task fills it after 10ms.
        item = await q.get()
        assert item == "hello"
        await producer

    asyncio.run(_go())


def test_open_text_writer_round_trip() -> None:
    """``open_text_writer`` writes UTF-8 text on both runtimes; the
    file round-trips through ``open(path).read()``."""
    p = c.make_scratch_path(prefix="cpy-writer-", suffix=".txt")
    try:
        with c.open_text_writer(p) as fh:
            fh.write("héllo\nwörld\n")
        with open(p, encoding="utf-8") as fh:
            assert fh.read() == "héllo\nwörld\n"
    finally:
        os.remove(p)


def test_log_contextvars_round_trip() -> None:
    """The structlog-bind helpers survive a bind/get/unbind round-trip
    on CPython. On MP they're no-ops (covered there); here we exercise
    the real-structlog branch."""
    import structlog

    # Start clean so this test doesn't pick up bindings from any
    # earlier test that forgot to unbind.
    structlog.contextvars.clear_contextvars()

    assert c.get_log_contextvars() == {}
    c.bind_log_contextvars(req_id="r-7", trace="t-9")
    snapshot = c.get_log_contextvars()
    assert snapshot["req_id"] == "r-7"
    assert snapshot["trace"] == "t-9"
    c.unbind_log_contextvars("req_id")
    assert "req_id" not in c.get_log_contextvars()
    c.unbind_log_contextvars("trace")
    assert c.get_log_contextvars() == {}


def test_async_queue_round_trip() -> None:
    """``make_async_queue`` returns a real ``asyncio.Queue`` on
    CPython. The ``_AsyncQueue`` class is the MicroPython fallback;
    construct + drive it directly here so the methods are exercised
    on CPython too — same pattern as ``_NoopLock`` / ``_StubLogger``."""
    import asyncio

    q = c.make_async_queue()

    async def _round_trip() -> None:
        await q.put("a")
        q.put_nowait("b")
        assert await q.get() == "a"
        assert await q.get() == "b"

    asyncio.run(_round_trip())

    # ── _AsyncQueue (MP fallback) ───────────────────────────────
    fallback = c._AsyncQueue()
    assert fallback.empty() is True
    assert fallback.qsize() == 0

    async def _fallback_round_trip() -> None:
        await fallback.put(1)
        fallback.put_nowait(2)
        assert fallback.qsize() == 2
        assert fallback.empty() is False
        assert await fallback.get() == 1
        assert await fallback.get() == 2
        assert fallback.empty() is True

    asyncio.run(_fallback_round_trip())


class TestAiterCompat:
    """``aiter_compat`` adapts sync OR async generators to the
    ``async for`` protocol. CPython's branch wraps real async
    generators (``__aiter__`` path); the sync-fallback branch
    matters for MicroPython callers but must still be exercised
    here so the CPython coverage gate sees both."""

    @pytest.mark.asyncio
    async def test_async_generator_passes_through(self) -> None:
        """``async def`` + ``yield`` produces an async generator on
        CPython — ``aiter_compat`` should detect ``__aiter__`` and
        delegate to the async iter protocol."""
        from exoclaw._compat import aiter_compat

        async def _agen() -> AsyncIterator[str]:
            yield "a"
            yield "b"

        out = []
        async for item in aiter_compat(_agen()):
            out.append(item)
        assert out == ["a", "b"]

    @pytest.mark.asyncio
    async def test_class_based_async_iter_delegates(self) -> None:
        """A class implementing ``__aiter__`` / ``__anext__`` is left
        untouched — ``aiter_compat`` just reuses its protocol."""
        from exoclaw._compat import aiter_compat

        class _ClassIter:
            def __init__(self) -> None:
                self._i = 0

            def __aiter__(self) -> "_ClassIter":
                return self

            async def __anext__(self) -> int:
                if self._i >= 3:
                    raise StopAsyncIteration
                self._i += 1
                return self._i

        out = []
        async for item in aiter_compat(_ClassIter()):
            out.append(item)
        assert out == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_sync_iterable_translates_stop_iteration(self) -> None:
        """A plain sync iterable (the shape MP produces from
        ``async def`` + ``yield``) gets bridged by translating
        ``StopIteration`` to ``StopAsyncIteration``."""
        from exoclaw._compat import aiter_compat

        out = []
        async for item in aiter_compat(iter([10, 20, 30])):
            out.append(item)
        assert out == [10, 20, 30]


class TestCPythonShimDelegations:
    """Cover the CPython branches of cross-runtime helpers — they
    delegate to stdlib so the tests are mostly "the call doesn't
    explode." Without these the gate counts these one-liners
    against the threshold."""

    def test_rmtree_removes_directory_tree(self, tmp_path: pathlib.Path) -> None:
        from exoclaw._compat import rmtree

        target = tmp_path / "tree"
        (target / "a" / "b").mkdir(parents=True)
        (target / "a" / "b" / "leaf.txt").write_text("x")
        rmtree(target)
        assert not target.exists()

    def test_platform_summary_returns_descriptive_string(self) -> None:
        from exoclaw._compat import platform_summary

        out = platform_summary()
        # CPython branch should mention Python version.
        assert "Python" in out

    def test_guess_image_mime_recognises_image_extensions(self) -> None:
        from exoclaw._compat import guess_image_mime

        assert guess_image_mime("/tmp/foo.png") == "image/png"
        assert guess_image_mime("/tmp/foo.jpeg") == "image/jpeg"

    def test_guess_image_mime_returns_none_for_non_images(self) -> None:
        """Copilot review fix — CPython branch was leaking
        non-image MIMEs (``text/plain`` etc.) because
        ``mimetypes.guess_type`` returns whatever it finds."""
        from exoclaw._compat import guess_image_mime

        assert guess_image_mime("/tmp/foo.txt") is None
        assert guess_image_mime("/tmp/foo.py") is None

    def test_which_delegates_to_shutil(self) -> None:
        from exoclaw._compat import which

        # ``sh`` exists on every CI box we run on.
        assert which("sh") is not None
        assert which("definitely-not-a-real-binary-xxxyyyzzz") is None

    def test_is_executable_delegates_to_os_access(self) -> None:
        from exoclaw._compat import is_executable

        assert is_executable("/bin/sh") is True
        assert is_executable("/definitely/not/a/real/path") is False
