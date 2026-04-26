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
import secrets
import sys
import tempfile
import threading
from collections.abc import AsyncIterator

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
