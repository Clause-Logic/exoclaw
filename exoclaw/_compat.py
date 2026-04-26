"""CPython / MicroPython runtime compatibility shims.

Built so exoclaw core's hot path can run on either runtime. CPython
gets the full stdlib (``ContextVar``, ``inspect``, ``secrets``,
``threading``, ``tempfile``) and ``structlog`` when available;
MicroPython gets bare-bones equivalents that preserve the call
contract on the one supported asyncio task.

Detection happens once at import time via
``sys.implementation.name``. Callers import the named helpers
directly and don't need to branch on the runtime themselves.

**What's intentionally NOT in here:**

- ``asyncio`` itself. ``uasyncio`` is a runtime-side subset; the
  shape of ``await`` / ``Task`` / ``Queue`` / ``Event`` is close
  enough that exoclaw's existing ``asyncio.X`` references work
  unchanged. Things like ``ContextVar`` that uasyncio doesn't
  ship are exposed via this module instead.
- Anything that isn't actually used in core. The shim only covers
  what the core import graph actually touches; if a future module
  needs (e.g.) ``hashlib``, add the shim then.

The ``IS_MICROPYTHON`` flag is exported for callers that need to
tell apart paths in their own logic — but most of the time, just
import the helper and it'll do the right thing.
"""

from __future__ import annotations

import os
import sys
from typing import Any

IS_MICROPYTHON: bool = sys.implementation.name == "micropython"


# ── Random bytes ─────────────────────────────────────────────────────────────

if IS_MICROPYTHON:  # pragma: no cover (cpython)
    # MicroPython has ``os.urandom`` but no ``secrets`` module. The
    # quality is the same — both delegate to the platform's CSPRNG.
    #
    # Per-runtime pragma protocol throughout this file:
    #   ``# pragma: no cover (cpython)``  — body unreachable on
    #     CPython, excluded from coverage.py.
    #   ``# pragma: no cover (micropython)``  — body unreachable
    #     on MicroPython, excluded from the runner's report.
    # See ``[tool.coverage.report]`` in ``pyproject.toml`` and
    # ``tests/test_micropython_runner.py::_executable_lines_for``.
    random_bytes = os.urandom
else:  # pragma: no cover (micropython)
    from secrets import token_bytes as random_bytes


# ── Locks ────────────────────────────────────────────────────────────────────


class _NoopLock:
    """No-op lock for single-threaded MicroPython.

    Same context-manager + acquire/release shape as
    ``threading.Lock`` so callers don't need a branch. uasyncio is
    cooperative single-threaded, so there's nothing to synchronize
    against — the lock is just a structural placeholder for code
    that was written defensively for CPython's threading model.
    """

    def __enter__(self) -> "_NoopLock":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def acquire(self, *args: object, **kwargs: object) -> bool:
        return True

    def release(self) -> None:
        return None


def make_lock() -> Any:
    """Return a ``threading.Lock``-shaped object.

    On CPython this is a real ``threading.Lock`` — exoclaw's
    monotonic-clock guard in ``_uuid7`` needs a true mutex because
    background threads (DBOS workers, structlog binding, etc.) can
    call into it concurrently. On MicroPython the cooperative
    single-task model means a no-op is correct: there's no
    pre-emption point inside the critical sections.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return _NoopLock()
    from threading import Lock

    return Lock()


# ── Task-local state (mirrors ``ContextVar`` API) ────────────────────────────

if IS_MICROPYTHON:  # pragma: no cover (cpython)
    _UNSET: Any = object()

    class _Token:
        """Reset token. Mirrors the opaque ``contextvars.Token`` shape
        — callers pass it to ``TaskLocal.reset``."""

        __slots__ = ("var", "old_value")

        def __init__(self, var: "TaskLocal", old_value: Any) -> None:
            self.var = var
            self.old_value = old_value

    class TaskLocal:
        """Single-task fallback for ``ContextVar``.

        uasyncio runs all tasks on one OS thread with cooperative
        scheduling; module-level state is safe across awaits as
        long as no two tasks observe each other's writes between
        ``set`` and ``reset``. exoclaw's call sites all wrap the
        ``set`` / ``reset`` pair around a single async-method body
        (no concurrent writers within one body), so a per-task
        snapshot isn't needed in practice on micro.

        If a future use case needs per-task isolation on micro,
        switch to a ``dict`` keyed by ``id(uasyncio.current_task())``
        with explicit cleanup in ``reset``. The ``Token`` interface
        already supports it.
        """

        __slots__ = ("_name", "_default", "_value")

        def __init__(self, name: str, *, default: Any = _UNSET) -> None:
            self._name = name
            self._default = default
            self._value: Any = _UNSET

        def get(self, default: Any = _UNSET) -> Any:
            if self._value is not _UNSET:
                return self._value
            if default is not _UNSET:
                return default
            if self._default is not _UNSET:
                return self._default
            raise LookupError(self._name)

        def set(self, value: Any) -> _Token:
            old = self._value
            self._value = value
            return _Token(self, old)

        def reset(self, token: _Token) -> None:
            self._value = token.old_value

else:  # pragma: no cover (micropython)
    # Re-export the real ContextVar so callers can use the shim
    # name. ``ContextVar`` is parameterised (``ContextVar[T]``);
    # the alias preserves that for static type checking.
    from contextvars import ContextVar as TaskLocal


# ── Inspect helpers ──────────────────────────────────────────────────────────

if IS_MICROPYTHON:  # pragma: no cover (cpython)

    def iscoroutinefunction(fn: Any) -> bool:
        """Detect ``async def`` functions on MicroPython.

        On MP an ``async def`` compiles to a function whose
        ``__class__.__name__`` is ``"generator"`` (the underlying
        machinery is generator-based coroutines). Plain ``def``
        functions are ``"function"``. ``hasattr(__name__)`` guards
        against builtins / methods that don't expose a ``__class__``
        with a ``__name__`` — those couldn't be ``async def`` anyway.

        Falls back to ``False`` for non-callables (None passed in
        from ``getattr(..., default=None)``)."""
        if fn is None:
            return False
        cls = getattr(fn, "__class__", None)
        if cls is None:
            return False
        return getattr(cls, "__name__", "") == "generator"

    def isasyncgenfunction(fn: Any) -> bool:
        """See ``iscoroutinefunction``. Conservative fallback —
        async generators on MP can't be reliably distinguished from
        plain coroutines without ``co_flags`` / ``inspect``, so we
        return ``False`` and the streaming-tool / async-append
        paths simply don't activate. The inline path is the safe
        fallback and works on every runtime."""
        return False

    def isawaitable(value: Any) -> bool:
        """Check if ``value`` can be ``await``ed.

        ``__await__`` covers PEP 492 awaitables (coroutines, Futures,
        Tasks, anything with an ``__await__`` method). ``send`` covers
        the older generator-based coroutines that uasyncio still
        accepts internally. Together this matches the surface that
        ``inspect.isawaitable`` tests on CPython for the kinds of
        objects exoclaw actually passes through.
        """
        return hasattr(value, "__await__") or hasattr(value, "send")

else:  # pragma: no cover (micropython)
    from inspect import isasyncgenfunction, isawaitable, iscoroutinefunction


# ── Scratch file paths ───────────────────────────────────────────────────────


def make_scratch_path(prefix: str = "tmp-", suffix: str = "", dir: str | None = None) -> str:
    """Return a path to a freshly-created empty scratch file.

    Cross-runtime replacement for ``tempfile.mkstemp`` — MicroPython
    1.27 doesn't expose ``O_CREAT`` / ``O_EXCL`` / ``O_RDWR``, so the
    micro branch builds a path manually and creates a placeholder
    with ``open("w")``; the CPython branch delegates to ``tempfile``
    directly so it gets exclusive-create + Windows-safe path joining.
    The caller is responsible for unlinking when done — same
    lifetime contract as ``tempfile.mkstemp``.

    ``dir`` defaults to the platform tempdir on CPython, ``/tmp``
    on MicroPython. On flash-only MicroPython boards, pass an
    explicit writable directory (e.g. workspace root).
    """
    if not IS_MICROPYTHON:  # pragma: no cover (micropython)
        # CPython: ``tempfile.mkstemp`` handles cross-platform path
        # joining and uses an exclusive-create flag (``O_CREAT |
        # O_EXCL``) that closes the symlink / pre-create race that
        # ``open(path, "w")`` would leave open on a shared tmpdir.
        from tempfile import mkstemp

        fd, path = mkstemp(prefix=prefix, suffix=suffix, dir=dir)
        os.close(fd)
        return path

    # MicroPython branch: build the path manually with a 64-bit
    # random suffix (``random_bytes(8)``) and create a zero-byte
    # placeholder. ``"x"`` (exclusive create) isn't supported on
    # 1.27, so ``"w"`` is the best we can do — the suffix
    # dominates the race window enough for the cooperative
    # single-task target this branch runs on.
    if dir is None:  # pragma: no cover (cpython)
        base = "/tmp"
    else:  # pragma: no cover (cpython)
        base = dir
    base = base.rstrip("/") or "/"  # pragma: no cover (cpython)
    try:  # pragma: no cover (cpython)
        os.mkdir(base)
    except OSError:  # pragma: no cover (cpython)
        pass  # already exists or read-only; the open below will fail loudly
    rand = "".join("{:02x}".format(b) for b in random_bytes(8))  # pragma: no cover (cpython)
    sep = "" if base.endswith("/") else "/"  # pragma: no cover (cpython)
    path = base + sep + prefix + rand + suffix  # pragma: no cover (cpython)
    open(path, "w").close()  # pragma: no cover (cpython)
    return path  # pragma: no cover (cpython)


def open_text_writer(path: str) -> Any:
    """Return a writable text-mode file handle for ``path``.

    CPython opens with ``encoding="utf-8"`` + ``newline=""`` to keep
    byte counts in sync with on-disk size (no Windows CRLF
    translation). MicroPython's ``open`` doesn't accept those
    kwargs — its text mode is always UTF-8 and there's no newline
    translation, so a bare ``open(path, "w")`` matches CPython's
    explicit semantics on the platforms exoclaw targets.

    Used by the streaming-tool path in ``executor.py`` to drain
    async-iterator chunks into a per-turn scratch file. Caller
    closes the handle; ``make_scratch_path`` already created the
    file as a 0-byte placeholder so this just opens it for write.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return open(path, "w")
    return open(path, "w", encoding="utf-8", newline="")  # pragma: no cover (micropython)


# ── Logger ───────────────────────────────────────────────────────────────────


class _StubLogger:
    """Minimal ``structlog``-compatible logger for MicroPython.

    Drops contextvars binding (no-op) and serializes one event per
    call. structured-log consumers downstream can grep the JSON
    lines just like the real structlog output, minus the
    contextvars-derived fields.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        import json as _json

        try:
            # Avoid ``{"a": 1, **fields}`` — PEP 448 dict unpacking
            # in dict literals isn't supported on MicroPython <=1.27.
            # Building via ``dict()`` + update keeps both runtimes happy.
            payload = {"level": level, "event": event, "logger": self._name}
            payload.update(fields)
            # ``json.dumps`` on MicroPython doesn't accept
            # ``ensure_ascii``. The default is True (escape non-ASCII)
            # on both runtimes — fine for log-line output.
            print(_json.dumps(payload))
        except Exception:
            print(f"[{level}] {event}")

    def info(self, event: str, **kw: Any) -> None:
        self._emit("info", event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._emit("warning", event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._emit("error", event, **kw)

    def exception(self, event: str, **kw: Any) -> None:
        self._emit("error", event, **kw)

    def debug(self, event: str, **kw: Any) -> None:
        self._emit("debug", event, **kw)

    def bind(self, **kw: Any) -> "_StubLogger":
        return self


def get_logger(name: str = "") -> Any:
    """Return a structlog-compatible logger.

    On CPython with ``structlog`` installed, returns a real
    ``structlog`` logger. On MicroPython or any environment
    without ``structlog``, returns a ``_StubLogger`` that prints
    structured JSON to stdout.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return _StubLogger(name)
    try:  # pragma: no cover (micropython)
        import structlog

        return structlog.get_logger(name)
    except ImportError:  # pragma: no cover
        return _StubLogger(name)


# ── Monotonic clock + path helpers (stdlib gaps on MicroPython) ──────────────


def monotonic_ms() -> int:
    """Return a monotonic timestamp in integer milliseconds.

    CPython has ``time.monotonic()`` returning float seconds; MP's
    ``time`` module exposes ``ticks_ms()`` instead (already int ms,
    monotonic but wraps at platform-specific limit). exoclaw only
    uses this for turn-duration measurement, so the wrap-around isn't
    a real concern — turns are seconds-long, the wrap window is
    minutes (ESP32) to hours (CPython int math).
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        import time as _time

        # ``ticks_ms`` is MP-only — opaque to ty, hence the getattr.
        return getattr(_time, "ticks_ms")()
    import time as _time  # pragma: no cover (micropython)

    return int(_time.monotonic() * 1000)  # pragma: no cover (micropython)


def monotonic_diff_ms(later: int, earlier: int) -> int:
    """Subtract two ``monotonic_ms()`` values safely.

    On MP, ``ticks_diff`` handles the wrap-around correctly when
    subtracting raw ``ticks_ms`` values. On CPython, the values are
    monotonically-increasing ints so plain subtraction works."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        import time as _time

        # ``ticks_diff`` is MP-only — opaque to ty, hence the getattr.
        return getattr(_time, "ticks_diff")(later, earlier)
    return later - earlier  # pragma: no cover (micropython)


def path_exists(path: str) -> bool:
    """Return True if ``path`` exists. ``os.path`` isn't a module on
    MicroPython (only ``os`` itself ships); ``os.stat`` raises if the
    path is missing, so we catch the ``OSError`` and translate."""
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def path_basename(path: str) -> str:
    """Return the trailing component of ``path``. Mirrors
    ``os.path.basename`` for the simple case exoclaw needs (no
    Windows separators — exoclaw never builds Windows paths through
    this helper). MP doesn't ship ``os.path``."""
    # Find the last separator and slice. Returns the full string when
    # no separator is present (mirrors stdlib behaviour).
    idx = path.rfind("/")
    if idx == -1:
        return path
    return path[idx + 1 :]


# ── Async queue (uasyncio fallback for ``asyncio.Queue``) ────────────────────


class _AsyncQueue:
    """Minimal ``asyncio.Queue`` replacement for MicroPython.

    uasyncio doesn't ship ``Queue`` (the unix port's frozen
    ``asyncio`` build has ``Lock``, ``Event``, ``Task``, ``gather``,
    ``wait_for``, but no ``Queue``). exoclaw's bus only uses
    ``put`` / ``get`` so the surface stays tiny — backed by a list
    and an ``Event`` that ``put`` sets and ``get`` clears when the
    list empties.

    Cooperative-single-task semantics: there's no preemption point
    inside ``put`` / ``get`` between checking and mutating the list,
    so a plain list is safe without a lock.
    """

    def __init__(self) -> None:
        self._items: list[Any] = []
        # Lazy import — module-level import would force every CPython
        # caller to pay for asyncio import even if they don't use the
        # queue (and we want this class importable even when asyncio
        # isn't installed yet during ``import exoclaw._compat``).
        import asyncio as _asyncio

        self._event = _asyncio.Event()

    def qsize(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return not self._items

    async def put(self, item: Any) -> None:
        self._items.append(item)
        self._event.set()

    def put_nowait(self, item: Any) -> None:
        self._items.append(item)
        self._event.set()

    async def get(self) -> Any:
        while not self._items:
            await self._event.wait()
            self._event.clear()
        return self._items.pop(0)


def make_async_queue() -> Any:
    """Return an ``asyncio.Queue``-shaped object for the running runtime.

    On CPython this is a real ``asyncio.Queue`` (full feature set,
    backpressure via ``maxsize``, etc.). On MicroPython this is the
    minimal ``_AsyncQueue`` defined above — same ``put`` / ``get``
    surface, no ``maxsize`` because exoclaw doesn't use that anywhere.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return _AsyncQueue()
    import asyncio  # pragma: no cover (micropython)

    return asyncio.Queue()  # pragma: no cover (micropython)


# ── Logger contextvars (per-turn binding) ────────────────────────────────────


def get_log_contextvars() -> dict:
    """Return the current per-task structlog contextvars snapshot.

    On CPython delegates to ``structlog.contextvars.get_contextvars``;
    on MicroPython returns an empty dict (no per-task binding —
    cooperative single-task model means there's nothing to scope).
    Callers use this to capture-then-restore around nested binds.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return {}
    try:  # pragma: no cover (micropython)
        import structlog.contextvars as _cv

        return dict(_cv.get_contextvars())
    except ImportError:  # pragma: no cover
        return {}


def bind_log_contextvars(**kw: Any) -> None:
    """Bind keys into the current per-task structlog contextvars.

    No-op on MicroPython — there's no contextvars layer to bind into,
    and the ``_StubLogger`` doesn't carry per-task state. The cost of
    losing this on micro is observability fields disappear from log
    lines; the cost of NOT no-op'ing it is a runtime error on every
    turn. Trade-off matches the scope of the shim.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return
    try:  # pragma: no cover (micropython)
        import structlog.contextvars as _cv

        _cv.bind_contextvars(**kw)
    except ImportError:  # pragma: no cover
        pass


def unbind_log_contextvars(*keys: str) -> None:
    """Unbind keys from the current per-task structlog contextvars.

    No-op on MicroPython — see ``bind_log_contextvars``.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return
    try:  # pragma: no cover (micropython)
        import structlog.contextvars as _cv

        _cv.unbind_contextvars(*keys)
    except ImportError:  # pragma: no cover
        pass


__all__ = [
    "IS_MICROPYTHON",
    "TaskLocal",
    "bind_log_contextvars",
    "get_log_contextvars",
    "get_logger",
    "isasyncgenfunction",
    "isawaitable",
    "iscoroutinefunction",
    "make_async_queue",
    "make_lock",
    "make_scratch_path",
    "monotonic_diff_ms",
    "monotonic_ms",
    "open_text_writer",
    "path_basename",
    "path_exists",
    "random_bytes",
    "unbind_log_contextvars",
]
