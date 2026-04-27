"""MicroPython side of the compat-shim test matrix.

Runs under ``tests/_micropython_runner/run.py``. Pure-Python — no
``pytest`` / fixture machinery (MicroPython doesn't have it). Each
``test_*`` function is invoked with no args; failures raise.

Mirrors the assertions in ``tests/test_compat.py`` (the CPython
side) but on the MicroPython runtime: same surface, opposite
branches. Coverage of ``_compat.py`` is tracked by the runner via
``sys.settrace``; the CPython-side wrapper
(``tests/test_micropython_runner.py``) parses the report and
asserts coverage ≥ 95% of MicroPython-reachable lines.
"""

import os

# The runner stages the full ``exoclaw/`` package under ``tmp_path``,
# so ``import exoclaw._compat`` works the same way it does on CPython.
from exoclaw import _compat as c


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


def test_isawaitable_distinguishes_awaitables():
    """``isawaitable`` returns True for coroutines / generators
    (anything with ``__await__`` or ``send``) and False for plain
    values."""

    async def _coro():
        return 1

    awaitable = _coro()
    assert c.isawaitable(awaitable) is True
    # Plain value: not awaitable.
    assert c.isawaitable(42) is False
    assert c.isawaitable("hi") is False


def test_iscoroutinefunction_handles_none():
    """``getattr(tool, 'execute_streaming', None)`` returns ``None``
    when the attribute is missing — ``iscoroutinefunction(None)``
    must safely return False rather than crashing."""
    assert c.iscoroutinefunction(None) is False


def test_decode_utf8_lossy_handles_truncated_codepoint():
    """Truncated multi-byte UTF-8 at the tail (``e2 98`` is the
    start of the 3-byte ★ codepoint without its third byte) →
    return the prefix that DOES decode cleanly. CPython does this
    via ``errors='ignore'``; MP can't pass that kwarg, so the
    helper trims trailing bytes one at a time until a valid
    decode."""
    # ``hello\xe2\x98`` — valid prefix + truncated codepoint.
    truncated = b"hello\xe2\x98"
    out = c.decode_utf8_lossy(truncated)
    assert out == "hello"


def test_decode_utf8_lossy_handles_clean_input():
    """Round-trip a clean UTF-8 byte string."""
    assert c.decode_utf8_lossy("héllo".encode("utf-8")) == "héllo"


def test_decode_utf8_lossy_handles_corrupt_input():
    """Bytes that fail decode at all four trim levels fall back
    to ASCII-with-question-mark. Pathological input — exoclaw's
    streaming-tool path won't actually produce this."""
    # Four high-bytes in a row never form a valid UTF-8 codepoint
    # tail — the four-trim window can't recover.
    out = c.decode_utf8_lossy(b"\x80\x80\x80\x80")
    # Either clean ASCII fallback or empty — both prove no crash.
    assert isinstance(out, str)


def test_stub_logger_falls_back_on_unencodable_field():
    """The stub logger's ``json.dumps`` wraps in try/except and
    falls back to ``[level] event`` on any exception (e.g. an
    unencodable field). Verifies the fallback path runs without
    crashing."""
    log = c._StubLogger("upy-fallback")
    # An ``object()`` instance isn't JSON-serializable — forces the
    # fallback path. We don't capture stdout on MP (no
    # ``contextlib.redirect_stdout``); just verify no exception.
    log.info("err", bad=object())


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
