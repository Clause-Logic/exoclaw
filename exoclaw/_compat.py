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
        """Conservative MicroPython fallback.

        ``inspect`` on MicroPython is stub-only and ``co_flags``
        introspection isn't reliable across builds. Returning
        ``False`` keeps every opt-in capability detection on the
        inline path — which is the safe fallback because the
        inline path is always supported. Streaming-tool / async-
        append paths just don't activate on micro until/unless a
        marker-based detection is added.
        """
        return False

    def isasyncgenfunction(fn: Any) -> bool:
        """See ``iscoroutinefunction``. Same conservative fallback."""
        return False

else:  # pragma: no cover (micropython)
    from inspect import isasyncgenfunction, iscoroutinefunction


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


__all__ = [
    "IS_MICROPYTHON",
    "TaskLocal",
    "get_logger",
    "isasyncgenfunction",
    "iscoroutinefunction",
    "make_lock",
    "make_scratch_path",
    "random_bytes",
]
