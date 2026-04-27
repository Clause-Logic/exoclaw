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


# ── Path shim coverage ───────────────────────────────────────────


def test_path_construction_and_join():
    """Path joining via ``/`` and varargs construction. The
    constructor strips leading ``/`` from non-first parts and
    collapses repeated slashes."""
    p = c.Path("/tmp", "exoclaw", "test")
    assert str(p) == "/tmp/exoclaw/test"

    # ``/`` operator returns a new Path
    p2 = c.Path("/tmp") / "child"
    assert str(p2) == "/tmp/child"

    # Repeated slashes collapse
    p3 = c.Path("/tmp//", "//child")
    assert str(p3) == "/tmp/child"

    # Empty input → "."
    p4 = c.Path()
    assert str(p4) == "."

    # Path-from-Path is the identity copy
    p5 = c.Path(p2)
    assert str(p5) == str(p2)

    # __repr__ wraps the string
    assert "Path(" in repr(p2)


def test_path_eq_hash_fspath():
    """__eq__ matches by string; __hash__ + __fspath__ work."""
    a = c.Path("/tmp/x")
    b = c.Path("/tmp/x")
    assert a == b
    # Comparison with a non-Path object returns False (no error).
    assert (a == "/tmp/x") is False
    assert hash(a) == hash(b)
    assert a.__fspath__() == "/tmp/x"


def test_path_parent_name_stem_suffix():
    """parent / name / stem / suffix derive from the string path
    without touching disk."""
    p = c.Path("/tmp/exoclaw/SKILL.md")
    assert str(p.parent) == "/tmp/exoclaw"
    assert p.name == "SKILL.md"
    assert p.stem == "SKILL"
    assert p.suffix == ".md"

    # No-extension case
    plain = c.Path("/tmp/README")
    assert plain.stem == "README"
    assert plain.suffix == ""

    # Root
    root = c.Path("/")
    assert str(root.parent) == "/"

    # Bare name
    bare = c.Path("name")
    assert str(bare.parent) == "."


def test_path_filesystem_ops(tmp_path=None):
    """exists/is_file/is_dir/mkdir/read_text/write_text/read_bytes/
    write_bytes/unlink/iterdir round-trip."""
    base = c.Path("/tmp", "upy-path-{}".format(id(c.Path)))
    # Clean any prior leftover (test re-run safety).
    if base.exists():
        c.rmtree(str(base))

    # mkdir(parents=True, exist_ok=True)
    inner = base / "child" / "grand"
    inner.mkdir(parents=True, exist_ok=True)
    assert inner.exists()
    assert inner.is_dir()
    assert not inner.is_file()
    # Re-mkdir with exist_ok is a no-op
    inner.mkdir(exist_ok=True)

    # write_text / read_text
    f = inner / "hello.txt"
    f.write_text("héllo", encoding="utf-8")
    assert f.exists()
    assert f.is_file()
    assert not f.is_dir()
    assert f.read_text() == "héllo"

    # write_bytes / read_bytes
    fb = inner / "bin.dat"
    n = fb.write_bytes(b"\x00\x01\x02")
    assert n == 3
    assert fb.read_bytes() == b"\x00\x01\x02"

    # iterdir
    names = sorted(p.name for p in inner.iterdir())
    assert names == ["bin.dat", "hello.txt"]

    # glob *.txt
    matches = sorted(str(p) for p in inner.glob("*.txt"))
    assert any(m.endswith("hello.txt") for m in matches)

    # glob non-pattern (literal child)
    lit = list(inner.glob("hello.txt"))
    assert len(lit) == 1

    # iterdir on a missing dir is a clean empty iterator
    missing = c.Path("/tmp/upy-does-not-exist-{}".format(id(c.Path)))
    assert list(missing.iterdir()) == []

    # unlink with missing_ok
    f.unlink()
    assert not f.exists()
    f.unlink(missing_ok=True)

    # rmtree wipes the whole subtree
    c.rmtree(str(base))
    assert not base.exists()


def test_path_expanduser_resolve_relative_to():
    """expanduser respects $HOME; resolve is a no-op; relative_to
    strips a prefix or raises."""
    # MP doesn't ship ``os.environ``; ``os.getenv`` is the cross-port
    # API. ``Path.expanduser`` reads ``HOME`` via ``os.getenv``;
    # under unix-port MP HOME is set by the parent shell, so we just
    # exercise the path-rewrite logic on whatever HOME is.
    home = os.getenv("HOME") or "/"
    e = c.Path("~/cfg").expanduser()
    assert str(e) == home.rstrip("/") + "/cfg" or str(e) == home + "/cfg"
    # Non-tilde paths pass through unchanged
    assert str(c.Path("/abs").expanduser()) == "/abs"

    # resolve is the same Path on MP
    p = c.Path("/tmp/x")
    assert p.resolve() is p

    # relative_to strips prefix
    rel = c.Path("/tmp/x/y").relative_to("/tmp")
    assert str(rel) == "x/y"

    raised = False
    try:
        c.Path("/tmp/x").relative_to("/etc")
    except ValueError:
        raised = True
    assert raised


def test_weak_value_dictionary_is_dict_on_mp():
    """On MP, ``WeakValueDictionary`` is a plain ``dict`` subclass.
    Plugin per-key lock maps work via ``setdefault``."""
    d = c.WeakValueDictionary()
    assert isinstance(d, dict)
    obj = object()
    d.setdefault("k", obj)
    assert d["k"] is obj


def test_rmtree_handles_missing_path():
    """rmtree on a missing path is a no-op (matches shutil.rmtree
    when called via the public helper that swallows errors)."""
    # _mp_rmtree on a missing path should not crash.
    c.rmtree("/tmp/upy-missing-{}".format(id(c.Path)))


def test_platform_summary_returns_string():
    """``platform_summary()`` returns a non-empty descriptor."""
    s = c.platform_summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_guess_image_mime_extensions():
    """Extension-based MIME guess covers the formats exoclaw's
    image-attachment path supports."""
    assert c.guess_image_mime("/tmp/foo.png") == "image/png"
    assert c.guess_image_mime("/tmp/foo.JPG") == "image/jpeg"
    assert c.guess_image_mime("/tmp/foo.jpeg") == "image/jpeg"
    assert c.guess_image_mime("/tmp/foo.gif") == "image/gif"
    assert c.guess_image_mime("/tmp/foo.webp") == "image/webp"
    assert c.guess_image_mime("/tmp/foo.bmp") == "image/bmp"
    # Unknown extension → None
    assert c.guess_image_mime("/tmp/foo.txt") is None


def test_which_returns_none_on_micropython():
    """MP has no PATH; ``which`` always returns None."""
    assert c.which("python") is None
    assert c.which("doesnotexist123") is None


def test_is_executable_returns_false_on_micropython():
    """MP has no subprocesses; ``is_executable`` always returns False."""
    assert c.is_executable("/bin/sh") is False
    assert c.is_executable("/tmp/anything") is False


def test_path_skips_empty_parts():
    """``Path("a", "", "b")`` skips the empty middle — covers the
    ``if not s: continue`` branch in ``__init__``."""
    p = c.Path("/tmp", "", "x")
    assert str(p) == "/tmp/x"


def test_path_basename_with_no_separator():
    """``path_basename`` without a ``/`` returns the input unchanged."""
    assert c.path_basename("plain") == "plain"


def test_async_queue_size_and_put_nowait():
    """Cover the synchronous surface of ``_AsyncQueue``: ``qsize``
    and ``put_nowait`` (no-await variants of the put path)."""
    q = c.make_async_queue()
    assert q.qsize() == 0
    q.put_nowait("x")
    assert q.qsize() == 1
    assert q.empty() is False


def test_unbind_log_contextvars_without_bag():
    """``unbind_log_contextvars`` is a no-op when the per-task bag
    hasn't been bound yet."""
    # Fresh task storage (or at least no log-ctx bag for *this* var).
    c.unbind_log_contextvars("never_bound_key")
