"""Verify ``exoclaw._compat`` actually loads + works on MicroPython.

The point of the compat shim is empty without proof that it runs
on the alternate interpreter. This test subprocess-runs MicroPython's
unix port against a flat copy of ``_compat.py`` (so ``exoclaw/__init__.py``
isn't pulled in — the rest of core hasn't been migrated yet) and
exercises every public function in the shim.

**Test rig:**

- ``tests/_micropython_stubs/__future__.py`` — minimal stub so
  ``from __future__ import annotations`` resolves on MicroPython.
  Equivalent to ``mip install __future__`` from micropython-lib;
  vendored here so the test doesn't need network in CI.
- ``tests/_micropython_stubs/typing.py`` — minimal stub so
  ``from typing import X`` resolves. Same logic.
- A flat copy of ``_compat.py`` (named ``exoclaw_compat.py``) is
  placed next to the stubs at test time so MicroPython can import
  it without going through the rest of the package.

**Skip behaviour:** if no ``micropython`` binary is on ``PATH``,
the test skips with a clear message rather than failing. Install
locally with ``brew install micropython`` (macOS) or build the
unix port from source.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

_STUBS_DIR = Path(__file__).parent / "_micropython_stubs"
_COMPAT_SRC = Path(__file__).parent.parent / "exoclaw" / "_compat.py"


def _has_micropython() -> bool:
    return shutil.which("micropython") is not None


pytestmark = pytest.mark.skipif(
    not _has_micropython(),
    reason="micropython binary not on PATH; install via `brew install micropython`",
)


def _run_micropython(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Subprocess-run a MicroPython script with the compat shim and
    stubs available on ``MICROPYPATH``.

    ``MICROPYPATH`` is MicroPython's analogue of ``PYTHONPATH``;
    paths are colon-separated and a leading colon is significant
    (it inserts MicroPython's default search path at that position).
    """
    flat_dir = tmp_path / "upy"
    flat_dir.mkdir()
    # Copy the shim under a flat name — importing as ``exoclaw._compat``
    # would trigger ``exoclaw/__init__.py``, which still uses CPython-only
    # idioms in the rest of core. Once the package is fully migrated the
    # test can switch to the package-qualified import.
    shutil.copy(_COMPAT_SRC, flat_dir / "exoclaw_compat.py")
    # Also copy the __future__ / typing stubs alongside.
    for stub in _STUBS_DIR.glob("*.py"):
        shutil.copy(stub, flat_dir / stub.name)

    env_path = f":{flat_dir}"
    return subprocess.run(
        ["micropython", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={"MICROPYPATH": env_path, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"},
    )


def test_compat_module_imports_and_runs(tmp_path: Path) -> None:
    """Smoke-test the whole public surface of ``exoclaw._compat`` on
    MicroPython. One script exercises every exported helper; the
    test asserts exit code 0 and the final OK marker.
    """
    script = textwrap.dedent(
        """
        import os
        import exoclaw_compat as c

        assert c.IS_MICROPYTHON is True, "MicroPython detection failed"

        # random_bytes — should return ``n`` bytes from the platform CSPRNG.
        b = c.random_bytes(16)
        assert len(b) == 16
        assert isinstance(b, (bytes, bytearray))

        # make_lock — context manager and acquire/release shape.
        lock = c.make_lock()
        with lock:
            pass
        assert lock.acquire() is True
        lock.release()

        # TaskLocal — get / set / reset round-trip.
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

        # iscoroutinefunction / isasyncgenfunction — conservative
        # MicroPython fallbacks return False, keeping callers on the
        # inline path. (CPython would inspect __code__.co_flags; we
        # don't.)
        assert c.iscoroutinefunction(lambda: None) is False
        assert c.isasyncgenfunction(lambda: None) is False

        # make_scratch_path — creates a zero-byte file and returns
        # its path. Caller is responsible for unlinking.
        p = c.make_scratch_path(prefix="upy-test-", suffix=".txt", dir="/tmp")
        try:
            with open(p) as fh:
                assert fh.read() == ""
        finally:
            os.remove(p)

        # get_logger — returns a structlog-shaped logger that doesn't
        # crash on basic usage.
        log = c.get_logger("test")
        log.info("event_one", k=1)
        log.warning("event_two")
        log.error("event_three", exc="boom")

        print("OK")
        """
    ).strip()

    result = _run_micropython(script, tmp_path)
    assert result.returncode == 0, (
        f"micropython exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    # The script ends by printing ``OK`` after every assert passes.
    # The logger calls also print structured JSON before that, so
    # we look for OK on the last non-empty line.
    last = [line for line in result.stdout.splitlines() if line.strip()][-1]
    assert last == "OK", f"expected OK marker on last line, got: {last!r}"


def test_runs_on_cpython_too() -> None:
    """The shim isn't useful if it only works on MicroPython — the
    whole point is dual-runtime. Sanity-check every exported helper
    inside this CPython process. The branches for CPython are taken
    at import time via ``IS_MICROPYTHON``, so this is exercising the
    other half of the file.
    """
    from exoclaw import _compat as c

    assert c.IS_MICROPYTHON is False
    assert len(c.random_bytes(16)) == 16

    with c.make_lock():
        pass

    var = c.TaskLocal("cpython_test", default=None)
    token = var.set(99)
    assert var.get() == 99
    var.reset(token)

    async def _async_fn() -> None:
        return None

    async def _async_gen() -> "AsyncIterator[int]":
        yield 1

    assert c.iscoroutinefunction(_async_fn) is True
    assert c.iscoroutinefunction(lambda: None) is False
    assert c.isasyncgenfunction(_async_gen) is True
    assert c.isasyncgenfunction(_async_fn) is False

    import os

    p = c.make_scratch_path(prefix="cpy-test-", suffix=".txt")
    try:
        assert os.path.exists(p)
        assert os.path.getsize(p) == 0
    finally:
        os.remove(p)

    log = c.get_logger("test")
    # Real structlog logger has bind() that returns a new logger.
    bound = log.bind(extra="ok") if hasattr(log, "bind") else log
    bound.info("smoke_test_cpython")


# Skip flag won't import sys, so silence unused-import.
_ = sys
