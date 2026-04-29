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
    from threading import Lock  # pragma: no cover (micropython)

    return Lock()  # pragma: no cover (micropython)


# ── Task-local state (mirrors ``ContextVar`` API) ────────────────────────────

if IS_MICROPYTHON:  # pragma: no cover (cpython)
    _UNSET: Any = object()

    # Per-task storage map: ``{ id(task): { id(var): value } }``.
    #
    # MicroPython doesn't ship ``contextvars`` and uasyncio's ``Task``
    # is a C struct with no place for arbitrary attributes (the
    # ``data`` slot is reserved for queue-waiting state). The trick:
    # use ``id(asyncio.current_task())`` as the key into a
    # module-level dict. Each task sees its own inner dict; concurrent
    # tasks under the cooperative single-task scheduler get real
    # per-task isolation.
    #
    # Lifecycle: ``utils.tasks.create_isolated_task`` wraps the
    # caller's coroutine in a finally-block that pops the task's
    # entry on completion. The ``main`` task started by
    # ``asyncio.run`` doesn't go through that wrapper but there's
    # only one of those per process — bounded leak.
    _task_storage: dict = {}

    def _current_task_dict() -> dict:
        """Return the per-task value dict, creating if needed.

        Falls back to a single module-level dict when no task is
        running (e.g. import-time ``set`` calls — exoclaw doesn't
        have any but a defensive fallback keeps non-task callers
        from raising ``LookupError``)."""
        import asyncio as _asyncio

        try:
            task = _asyncio.current_task()
        except Exception:
            task = None
        if task is None:
            # Use a sentinel id (0) for the "no task" bucket — keeps
            # the data structure homogeneous.
            return _task_storage.setdefault(0, {})
        return _task_storage.setdefault(id(task), {})

    def _drop_current_task_storage() -> None:
        """Pop the current task's storage on exit.

        Called by ``utils.tasks.create_isolated_task``'s wrapper in
        a ``finally`` block, so the task's dict goes away when the
        task body completes (normal return, exception, or
        cancellation)."""
        import asyncio as _asyncio

        try:
            task = _asyncio.current_task()
        except Exception:
            return
        if task is not None:
            _task_storage.pop(id(task), None)

    class _Token:
        """Reset token. Mirrors the opaque ``contextvars.Token`` shape
        — callers pass it to ``TaskLocal.reset``."""

        __slots__ = ("var", "old_value")

        def __init__(self, var: "TaskLocal", old_value: Any) -> None:
            self.var = var
            self.old_value = old_value

    class TaskLocal:
        """Per-task value slot, mirrors ``contextvars.ContextVar``.

        Uses ``id(asyncio.current_task())`` as the storage key, so
        concurrent tasks under uasyncio's cooperative single-task
        scheduler each see their own value — real isolation, not
        the module-level "everyone shares" fallback an earlier
        version of this shim shipped.

        Cleanup happens at task end via the
        ``create_isolated_task`` wrapper in ``utils.tasks`` (it
        calls ``_drop_current_task_storage`` in a ``finally``).
        Ad-hoc ``asyncio.create_task`` consumers leak; exoclaw's
        codebase has a banned-import rule that funnels everything
        through ``create_isolated_task``."""

        __slots__ = ("_name", "_default")

        def __init__(self, name: str, *, default: Any = _UNSET) -> None:
            self._name = name
            self._default = default

        def get(self, default: Any = _UNSET) -> Any:
            d = _current_task_dict()
            if id(self) in d:
                return d[id(self)]
            if default is not _UNSET:
                return default
            if self._default is not _UNSET:
                return self._default
            raise LookupError(self._name)

        def set(self, value: Any) -> _Token:
            d = _current_task_dict()
            old = d.get(id(self), _UNSET)
            d[id(self)] = value
            return _Token(self, old)

        def reset(self, token: _Token) -> None:
            d = _current_task_dict()
            if token.old_value is _UNSET:
                d.pop(id(token.var), None)
            else:
                d[id(token.var)] = token.old_value

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
        """Conservative ``False`` on MicroPython.

        ``async def`` (coroutine) and ``async def`` with ``yield``
        (async generator) both compile to ``generator``-class
        functions on MP, share the same ``__code__`` shape, and
        have no ``co_flags`` to introspect — there's no static way
        to tell them apart without calling the function.

        Callers that need MP-side streaming-tool dispatch (the
        memory-model.md Step D path) shouldn't rely on this shim
        alone — ``executor.execute_tool_with_handle`` does
        per-runtime dispatch instead: ``async for`` on CPython
        (after ``inspect.isasyncgenfunction`` clears it pre-call)
        and plain ``for`` on MP (since MP collapses ``async def +
        yield`` to a sync generator). This shim is just the
        conservative pre-call probe — returning ``False`` keeps
        callers off any path that would assume static introspection
        worked."""
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


def decode_utf8_lossy(data: bytes) -> str:
    """Decode ``data`` as UTF-8, dropping incomplete multibyte
    sequences at the tail.

    CPython would do this via ``data.decode("utf-8", errors="ignore")``
    but MicroPython's ``bytes.decode`` doesn't accept the ``errors``
    keyword. Implement it by trimming trailing bytes one at a time
    until the decode succeeds — the only case exoclaw hits is a
    chunk-boundary cut mid-codepoint at the end of a buffer, and the
    longest UTF-8 codepoint is 4 bytes, so worst-case we trim 3.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        for trim in range(4):
            try:
                if trim == 0:
                    return data.decode("utf-8")
                return data[:-trim].decode("utf-8")
            except UnicodeError:
                continue
        # All four-byte windows raised — corrupt input. Return what
        # we can ASCII-decode.
        return "".join(chr(b) if b < 128 else "?" for b in data)
    return data.decode("utf-8", errors="ignore")  # pragma: no cover (micropython)


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

    Reads the per-task structlog-contextvars bag at emit time so log
    lines carry ``turn.id`` / ``session.key`` / etc. just like the
    CPython structlog path. The per-task isolation comes from
    ``TaskLocal``'s ``id(asyncio.current_task())`` keying — every
    concurrent turn writes its own bag without cross-talk.
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
            # Pull the per-task structlog contextvars bag — these
            # are the ``turn.id`` / ``session.key`` / ``channel`` /
            # etc. that ``bind_log_contextvars`` populated. Shipped
            # under the field names the CPython structlog setup uses,
            # so a downstream LogsQL query reads the same fields on
            # both runtimes.
            payload.update(get_log_contextvars())
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
        # CPython's structlog auto-captures ``sys.exc_info()`` inside
        # ``.exception``; the stub used in MP / structlog-less envs
        # didn't, so log lines emitted from ``except: log.exception(...)``
        # blocks dropped the actual exception entirely. Capture it
        # explicitly here so the chip surfaces the same diagnostic
        # info as the server.
        import sys as _sys

        exc_info = _sys.exc_info()
        if exc_info and exc_info[1] is not None:
            kw.setdefault("error", repr(exc_info[1]))
            self._print_exception_trace(exc_info)
        self._emit("error", event, **kw)

    def _print_exception_trace(self, exc_info: Any) -> None:
        # Split into a helper so the per-runtime branches are each
        # behind a single pragma — the inline if/else version had
        # the runner mark inner ``try``/``except`` lines as missing
        # because pragmas on the outer ``if`` don't propagate down.
        if IS_MICROPYTHON:  # pragma: no cover (cpython): MP-only branch
            import sys as _sys

            try:
                # ``sys.print_exception`` is MP-only — ``getattr`` keeps
                # ty quiet on CPython where the attribute genuinely
                # doesn't exist.
                _sys_print_exception = getattr(_sys, "print_exception")
                _sys_print_exception(exc_info[1])
            except Exception:
                pass
        else:  # pragma: no cover (micropython): CPython-only branch
            try:
                import traceback as _tb

                _tb.print_exception(exc_info[0], exc_info[1], exc_info[2])
            except Exception:
                pass

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


def getenv(name: str, default: str | None = None) -> str | None:
    """Cross-runtime ``os.getenv``. CPython and MicroPython's unix port
    both ship ``os.getenv``; MicroPython on bare-metal targets (ESP32 et
    al) does not, so a direct ``os.getenv`` call ``AttributeError``s and
    crashes the chip during init. This helper falls back to ``default``
    on those runtimes — chip code should keep providing a sensible
    default for any value it tries to read."""
    fn = getattr(os, "getenv", None)
    if fn is None:
        return default
    return fn(name, default)


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


class _AsyncSemaphore:
    """Minimal ``asyncio.Semaphore`` replacement for MicroPython.

    uasyncio ships ``Lock`` and ``Event`` but not ``Semaphore``
    (the unix port's frozen ``asyncio`` build is checked
    explicitly: ``hasattr(asyncio, 'Semaphore')`` is ``False``).
    Plugins that want a counting cap on concurrent tasks
    — exoclaw-subagent's ``AsyncioSpawner(max_concurrent=N)``
    is the load-bearing caller — would otherwise have to gate the
    feature on runtime, which contradicts the "any MP target, any
    policy" framing.

    Counter + ``asyncio.Event`` is enough: ``acquire`` waits for
    the event, decrements, and clears the event when the counter
    hits zero; ``release`` increments and sets the event. The
    cooperative single-task model on MP means the
    wait-then-decrement window is race-free without a real lock —
    no preemption point between ``await event.wait()`` returning
    and the counter check on the next line.

    Same ``acquire`` / ``release`` / ``__aenter__`` / ``__aexit__``
    surface as ``asyncio.Semaphore`` so plugins use ``async with
    sem:`` unchanged across both runtimes.
    """

    def __init__(self, value: int) -> None:
        # Match CPython's ``asyncio.Semaphore`` boundary contract —
        # ``value=0`` is valid (all callers wait until ``release``)
        # and only negatives raise.
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._value = value
        # Lazy import — same reasoning as ``_AsyncQueue``.
        import asyncio as _asyncio

        self._event = _asyncio.Event()
        if value > 0:
            self._event.set()

    async def acquire(self) -> bool:
        # Loop on spurious wake — ``release`` from a sibling task
        # may set the event between us seeing ``_value == 0`` and
        # ``acquire`` running again. Cooperative scheduling means
        # we won't actually loop more than necessary; this is just
        # defensive structure.
        while True:
            await self._event.wait()
            if self._value > 0:
                self._value -= 1
                if self._value == 0:
                    self._event.clear()
                return True

    def release(self) -> None:
        self._value += 1
        self._event.set()

    async def __aenter__(self) -> "_AsyncSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()


def make_semaphore(value: int) -> Any:
    """Return an ``asyncio.Semaphore``-shaped object for the running runtime.

    On CPython this is a real ``asyncio.Semaphore``. On MicroPython
    this is the minimal ``_AsyncSemaphore`` above — same
    ``acquire`` / ``release`` / ``async with`` surface so callers
    don't branch on runtime.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return _AsyncSemaphore(value)
    import asyncio  # pragma: no cover (micropython)

    return asyncio.Semaphore(value)  # pragma: no cover (micropython)


# ── Logger contextvars (per-turn binding) ────────────────────────────────────


if IS_MICROPYTHON:  # pragma: no cover (cpython)
    # Sentinel key into the per-task storage dict for the logger
    # contextvars bag. The bag is a sub-dict; callers bind/unbind
    # keys via ``bind_log_contextvars`` / ``unbind_log_contextvars``,
    # and ``_StubLogger._emit`` reads it to enrich log lines.
    _LOG_CTX_KEY = "__exoclaw_log_ctx__"


def get_log_contextvars() -> dict:
    """Return the current per-task structlog contextvars snapshot.

    On CPython delegates to ``structlog.contextvars.get_contextvars``.
    On MicroPython reads the per-task bag in ``_task_storage`` (real
    per-task isolation now — every concurrent turn gets its own
    bag, scoped by ``id(asyncio.current_task())``)."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        d = _current_task_dict()
        return dict(d.get(_LOG_CTX_KEY, {}))
    try:  # pragma: no cover (micropython)
        import structlog.contextvars as _cv

        return dict(_cv.get_contextvars())
    except ImportError:  # pragma: no cover
        return {}


def bind_log_contextvars(**kw: Any) -> None:
    """Bind keys into the current per-task structlog contextvars.

    On MicroPython this writes into the per-task bag — concurrent
    turns each get their own ``turn.id`` / ``session.key`` /
    etc. without trampling each other. Same observability story
    as CPython, just with our hand-rolled context layer."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        d = _current_task_dict()
        bag = d.get(_LOG_CTX_KEY)
        if bag is None:
            bag = {}
            d[_LOG_CTX_KEY] = bag
        bag.update(kw)
        return
    try:  # pragma: no cover (micropython)
        import structlog.contextvars as _cv

        _cv.bind_contextvars(**kw)
    except ImportError:  # pragma: no cover
        pass


def unbind_log_contextvars(*keys: str) -> None:
    """Unbind keys from the current per-task structlog contextvars.

    On MicroPython removes from the per-task bag; on CPython
    delegates to structlog. ``KeyError`` on missing keys is
    swallowed so partial unbind sets behave the same on both
    runtimes (matches structlog semantics)."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        d = _current_task_dict()
        bag = d.get(_LOG_CTX_KEY)
        if bag is None:
            return
        for k in keys:
            bag.pop(k, None)
        return
    try:  # pragma: no cover (micropython)
        import structlog.contextvars as _cv

        _cv.unbind_contextvars(*keys)
    except ImportError:  # pragma: no cover
        pass


# ── Pathlib shim ─────────────────────────────────────────────────────────────


if IS_MICROPYTHON:  # pragma: no cover (cpython)
    from typing import Iterator as _Iterator

    class Path:
        """Minimal ``pathlib.Path``-shaped class for MicroPython.

        Plugins that handle paths (workspace dirs, skill dirs, JSONL
        history, etc.) should import this from ``exoclaw._compat``
        instead of ``pathlib`` directly. CPython gets the real
        ``pathlib.Path``; MP gets this hand-rolled subset.

        Supports the operations exoclaw plugins use: ``/``, ``str()``,
        ``read_text``, ``write_text``, ``read_bytes``, ``write_bytes``,
        ``exists``, ``is_file``, ``is_dir``, ``mkdir``, ``parent``,
        ``name``, ``stem``, ``suffix``, ``expanduser``, ``resolve``,
        ``relative_to``, ``iterdir``, ``glob`` (``*.ext`` only),
        ``unlink``.

        ``encoding`` kwargs on ``read_text`` / ``write_text`` are
        accepted and ignored — MP's ``open`` is always UTF-8 in
        text mode."""

        __slots__ = ("_path",)

        def __init__(self, *parts: object) -> None:
            if len(parts) == 1 and isinstance(parts[0], Path):
                # ``ty`` sees ``Path`` as a union (this MP class +
                # ``pathlib.Path`` from the CPython branch) and can't
                # confirm ``_path`` lives on the union. The runtime
                # gate guarantees only the MP class is reachable here.
                self._path = parts[0]._path  # type: ignore[unresolved-attribute]
                return
            cleaned: list[str] = []
            for i, p in enumerate(parts):
                s = str(p)
                if not s:
                    continue
                if i > 0:
                    s = s.lstrip("/")
                cleaned.append(s.rstrip("/") if i < len(parts) - 1 else s)
            joined = "/".join(cleaned) if cleaned else ""
            while "//" in joined:
                joined = joined.replace("//", "/")
            self._path = joined or "."

        def __str__(self) -> str:
            return self._path

        def __repr__(self) -> str:
            return "Path({!r})".format(self._path)

        def __truediv__(self, other: object) -> "Path":
            return Path(self._path, str(other))

        def __eq__(self, other: object) -> bool:
            if isinstance(other, Path):
                # See ``__init__`` for why ty needs the override here —
                # the runtime gate keeps this branch MP-only.
                return self._path == other._path  # type: ignore[unresolved-attribute]
            return False

        def __hash__(self) -> int:
            return hash(self._path)

        def __fspath__(self) -> str:
            return self._path

        @property
        def parent(self) -> "Path":
            idx = self._path.rfind("/")
            if idx == -1:
                return Path(".")
            if idx == 0:
                return Path("/")
            return Path(self._path[:idx])

        @property
        def name(self) -> str:
            return self._path.split("/")[-1]

        @property
        def stem(self) -> str:
            n = self.name
            i = n.rfind(".")
            return n if i <= 0 else n[:i]

        @property
        def suffix(self) -> str:
            n = self.name
            i = n.rfind(".")
            return "" if i <= 0 else n[i:]

        def exists(self) -> bool:
            try:
                os.stat(self._path)
                return True
            except OSError:
                return False

        def is_file(self) -> bool:
            try:
                mode = os.stat(self._path)[0]
            except OSError:
                return False
            return not (mode & 0o040000)

        def is_dir(self) -> bool:
            try:
                mode = os.stat(self._path)[0]
            except OSError:
                return False
            return bool(mode & 0o040000)

        def mkdir(
            self,
            *,
            parents: bool = False,
            exist_ok: bool = False,
            mode: int = 0o777,
        ) -> None:
            if parents and not self.parent.exists():
                self.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.mkdir(self._path)
            except OSError:
                if not exist_ok or not self.is_dir():
                    raise

        def read_text(self, encoding: str | None = None) -> str:
            with open(self._path) as fh:
                return fh.read()

        def write_text(self, content: str, encoding: str | None = None) -> int:
            with open(self._path, "w") as fh:
                return fh.write(content)

        def read_bytes(self) -> bytes:
            with open(self._path, "rb") as fh:
                return fh.read()

        def write_bytes(self, content: bytes) -> int:
            with open(self._path, "wb") as fh:
                return fh.write(content)

        def expanduser(self) -> "Path":
            if self._path.startswith("~"):
                home = getenv("HOME", "/") or "/"
                return Path(home + self._path[1:])
            return self

        def resolve(self) -> "Path":
            # MP doesn't have ``realpath`` consistently. For exoclaw's
            # use case (display in system prompt), the input path is
            # already absolute or relative to a known root — resolving
            # to the same string is acceptable.
            return self

        def relative_to(self, other: "Path | str") -> "Path":
            other_s = str(other).rstrip("/")
            # Segment-boundary check — match either the exact path
            # or a prefix followed by ``/``. Bare ``startswith``
            # would accept ``/tmp/x``.relative_to(``/tmp/xy``) which
            # ``pathlib.Path`` rejects.
            ok = self._path == other_s or self._path.startswith(other_s + "/")
            if not ok:
                raise ValueError("{!r} not under {!r}".format(self._path, other_s))
            rest = self._path[len(other_s) :].lstrip("/")
            return Path(rest)

        def iterdir(self) -> "_Iterator[Path]":
            try:
                for nm in os.listdir(self._path):
                    yield Path(self._path, nm)
            except OSError:
                return

        def glob(self, pattern: str) -> "_Iterator[Path]":
            # Minimal glob — supports ``*.ext`` patterns only. Used
            # by plugin skill loaders for ``*.md`` style scans.
            if not pattern.startswith("*."):
                p = self / pattern
                if p.exists():
                    yield p
                return
            ext = pattern[1:]
            for child in self.iterdir():
                if child.name.endswith(ext):
                    yield child

        def unlink(self, missing_ok: bool = False) -> None:
            try:
                os.remove(self._path)
            except OSError:
                if not missing_ok:
                    raise

else:  # pragma: no cover (micropython)
    from pathlib import Path  # noqa: F401  (re-exported)


# ── WeakValueDictionary shim ─────────────────────────────────────────────────


if IS_MICROPYTHON:  # pragma: no cover (cpython)

    class WeakValueDictionary(dict):
        """Plain ``dict`` on MicroPython — ``weakref`` doesn't ship.

        Plugins use this as a per-key lock map (e.g. consolidation
        locks keyed by session). On a single-tenant chip the small
        bounded leak (one entry per key that ever existed) is
        negligible. Inherits from ``dict`` so ``setdefault``/
        ``__getitem__``/etc. work without extra method bodies."""

else:  # pragma: no cover (micropython)
    from weakref import WeakValueDictionary  # noqa: F401


# ── Recursive directory delete ───────────────────────────────────────────────


def rmtree(path: object) -> None:
    """Recursively delete ``path``. ``shutil.rmtree`` on CPython,
    hand-rolled walk-and-remove on MicroPython (which doesn't ship
    ``shutil``)."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        _mp_rmtree(str(path))
        return
    import shutil  # pragma: no cover (micropython)

    shutil.rmtree(str(path))  # pragma: no cover (micropython)


def _mp_rmtree(path: str) -> None:  # pragma: no cover (cpython)
    try:
        entries = os.listdir(path)
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass
        return
    for nm in entries:
        full = path.rstrip("/") + "/" + nm
        try:
            mode = os.stat(full)[0]
        except OSError:
            continue
        if mode & 0o040000:
            _mp_rmtree(full)
        else:
            try:
                os.remove(full)
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        pass


# ── Platform identifier ──────────────────────────────────────────────────────


def platform_summary() -> str:
    """Short runtime descriptor for system-prompt's ``Runtime`` block.

    CPython: ``"<system> <machine>, Python <version>"``;
    MicroPython: ``"<sysname> <machine>, MicroPython <version>"``
    via ``os.uname()`` (the only cross-port API).
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        try:
            uname = os.uname()
            sysname = getattr(uname, "sysname", "micropython")
            machine = getattr(uname, "machine", "")
            release = getattr(uname, "release", "")
            return "{} {}, MicroPython {}".format(sysname, machine, release).strip()
        except Exception:
            return "MicroPython"
    import platform  # pragma: no cover (micropython)

    system = platform.system()  # pragma: no cover (micropython)
    machine = platform.machine()  # pragma: no cover (micropython)
    py = platform.python_version()  # pragma: no cover (micropython)
    label = "macOS" if system == "Darwin" else system  # pragma: no cover (micropython)
    return f"{label} {machine}, Python {py}"  # pragma: no cover (micropython)


# ── Image MIME guess ─────────────────────────────────────────────────────────


_IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def guess_image_mime(path: str) -> str | None:
    """Extension-based MIME guess for the image-attachment path.

    Returns ``image/*`` for known image extensions, ``None`` for
    everything else. CPython delegates to ``mimetypes.guess_type``
    and filters to image-only so callers get the same answer as
    on MicroPython (where the lookup table is image-only by
    construction). Supported formats: ``png``, ``jpeg``, ``gif``,
    ``webp``, ``bmp``."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        path_lower = path.lower()
        for ext, mime in _IMAGE_MIME_BY_EXT.items():
            if path_lower.endswith(ext):
                return mime
        return None
    import mimetypes  # pragma: no cover (micropython)

    mime, _ = mimetypes.guess_type(path)  # pragma: no cover (micropython)
    if mime is None or not mime.startswith("image/"):  # pragma: no cover (micropython)
        return None
    return mime  # pragma: no cover (micropython)


# ── Executable lookup / X_OK shim ────────────────────────────────────────────


def which(binary: str) -> str | None:
    """Look up an executable on ``PATH``. ``shutil.which`` on CPython,
    ``None`` on MicroPython.

    MP on a microcontroller has no ``PATH`` and no subprocesses, so
    ``which`` always returns ``None``. Plugin skill loaders use this
    to decide whether a skill's ``requires.bins`` are met — on a
    chip the answer is always "no", which is correct."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return None
    import shutil  # pragma: no cover (micropython)

    return shutil.which(binary)  # pragma: no cover (micropython)


class aiter_compat:  # noqa: N801  — lower-case mirrors stdlib ``aiter``
    """Wrap a generator so ``async for`` works on both runtimes.

    CPython compiles ``async def f(): yield`` into an async
    generator with ``__aiter__`` / ``__anext__``. MicroPython 1.27
    compiles the same construct into a plain generator that has
    neither. Iterating the latter with ``async for`` raises
    ``AttributeError: 'generator' object has no attribute
    '__aiter__'`` on MP.

    ``aiter_compat(gen)`` adapts either shape:

    .. code-block:: python

        async for chunk in aiter_compat(_my_streaming_gen()):
            ...

    No-op overhead on CPython (passes through async ``__anext__``);
    on MicroPython, translates ``StopIteration`` from the
    underlying sync generator into ``StopAsyncIteration`` so the
    caller's ``async for`` exits cleanly."""

    __slots__ = ("_iter", "_is_async")

    def __init__(self, gen: Any) -> None:
        if hasattr(gen, "__aiter__"):
            self._iter = gen.__aiter__()
            self._is_async = True
        else:
            self._iter = iter(gen)
            self._is_async = False

    def __aiter__(self) -> "aiter_compat":
        return self

    async def __anext__(self) -> Any:
        if self._is_async:
            return await self._iter.__anext__()
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def is_executable(path: object) -> bool:
    """Check if ``path`` is executable. ``os.access(path, os.X_OK)``
    on CPython, always ``False`` on MicroPython (no subprocesses)."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return False
    return os.access(str(path), os.X_OK)  # pragma: no cover (micropython)


__all__ = [
    "IS_MICROPYTHON",
    "Path",
    "TaskLocal",
    "WeakValueDictionary",
    "aiter_compat",
    "bind_log_contextvars",
    "decode_utf8_lossy",
    "get_log_contextvars",
    "get_logger",
    "guess_image_mime",
    "is_executable",
    "isasyncgenfunction",
    "isawaitable",
    "iscoroutinefunction",
    "make_async_queue",
    "make_lock",
    "make_semaphore",
    "make_scratch_path",
    "monotonic_diff_ms",
    "monotonic_ms",
    "open_text_writer",
    "path_basename",
    "path_exists",
    "platform_summary",
    "random_bytes",
    "rmtree",
    "unbind_log_contextvars",
    "which",
]
