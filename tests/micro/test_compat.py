"""MicroPython side of the compat-shim test matrix.

Runs under ``tests/_micropython_runner/run.py``. Pure-Python — no
``pytest`` / fixture machinery (MicroPython doesn't have it). Each
``test_*`` function is invoked with no args; failures raise.

Mirrors the assertions in ``tests/test_micropython_compat.py::test_runs_on_cpython_too``
but on the MicroPython runtime: same surface, opposite branches.
Coverage of ``_compat.py`` is tracked by the runner via
``sys.settrace``; the CPython-side wrapper
(``tests/test_micropython_runner.py``) parses the report and
asserts coverage ≥ 95% of MicroPython-reachable lines.
"""

import os

# The runner sets up sys.path so ``_compat`` resolves to
# ``exoclaw/_compat.py`` (a flat copy or sym-link, since
# ``exoclaw/__init__.py`` still uses CPython-only imports).
import _compat as c


def test_branch_selection_micropython():
    """Prove the MicroPython branches in ``_compat.py`` were taken."""
    assert c.IS_MICROPYTHON is True, "MicroPython detection failed"

    # random_bytes ── micro path uses os.urandom (no ``secrets`` module).
    assert c.random_bytes is os.urandom

    # make_lock ── micro path returns _NoopLock (single-task uasyncio).
    assert isinstance(c.make_lock(), c._NoopLock)

    # TaskLocal ── micro is the hand-rolled class. ``contextvars``
    # doesn't exist on micro at all.
    try:
        import contextvars  # noqa: F401

        raise AssertionError("contextvars unexpectedly importable on micro")
    except ImportError:
        pass

    # iscoroutinefunction / isasyncgenfunction ── conservative
    # ``False`` fallbacks (CPython's ``inspect`` not available).
    assert c.iscoroutinefunction(lambda: None) is False
    assert c.isasyncgenfunction(lambda: None) is False

    # _StubLogger is the get_logger fallback on micro.
    assert isinstance(c.get_logger("test"), c._StubLogger)


def test_random_bytes_returns_correct_length():
    b = c.random_bytes(16)
    assert len(b) == 16
    assert isinstance(b, (bytes, bytearray))


def test_make_lock_context_manager():
    lock = c.make_lock()
    with lock:
        pass
    assert lock.acquire() is True
    assert lock.acquire(blocking=False) is True
    assert lock.release() is None


def test_task_local_get_set_reset():
    v = c.TaskLocal("test_var", default=None)
    assert v.get() is None  # default

    token = v.set(42)
    assert v.get() == 42

    token2 = v.set("hello")
    assert v.get() == "hello"

    v.reset(token2)
    assert v.get() == 42

    v.reset(token)
    assert v.get() is None


def test_task_local_lookup_error_when_no_default():
    v = c.TaskLocal("no_default")
    raised = False
    try:
        v.get()
    except LookupError:
        raised = True
    assert raised, "TaskLocal.get() should raise LookupError without default"


def test_task_local_get_with_explicit_default():
    v = c.TaskLocal("with_default")
    # Explicit default arg overrides the no-default lookup error.
    assert v.get(default="fallback") == "fallback"


def test_inspect_helpers_return_false():
    """Conservative MicroPython fallbacks: every callable inspects
    as not-async / not-async-generator. Keeps callers on the inline
    path."""
    assert c.iscoroutinefunction(lambda: None) is False
    assert c.iscoroutinefunction(test_inspect_helpers_return_false) is False
    assert c.isasyncgenfunction(lambda: None) is False


def test_make_scratch_path_creates_empty_file():
    p = c.make_scratch_path(prefix="upy-test-", suffix=".txt", dir="/tmp")
    try:
        with open(p) as fh:
            assert fh.read() == ""
    finally:
        os.remove(p)


def test_make_scratch_path_default_dir():
    """Default ``dir`` resolves to ``/tmp`` on MicroPython."""
    p = c.make_scratch_path(prefix="upy-default-")
    try:
        # Default base on micro is ``/tmp``.
        assert p.startswith("/tmp/upy-default-")
    finally:
        os.remove(p)


def test_stub_logger_emits_json():
    """Verify the stub logger doesn't crash on any of the level
    methods. (Output capture isn't trivial without
    ``contextlib.redirect_stdout`` so we just verify no exception.)"""
    log = c.get_logger("upy-test")
    log.debug("dbg_event", k=1)
    log.info("info_event", k=2)
    log.warning("warn_event")
    log.error("err_event", reason="x")
    log.exception("exc_event", exc="boom")
    # bind() is a no-op that returns self.
    assert log.bind(extra="ok") is log
