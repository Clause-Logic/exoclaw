"""Self-coverage for ``exoclaw.testing.assert_set_context_isolates_per_task``.

Two minimal dummy tools — one buggy (instance attr), one correct
(ContextVar) — exercise the helper end-to-end. The buggy tool must
trip the helper's assertion; the correct tool must pass.
"""

from __future__ import annotations

from contextvars import ContextVar

import pytest

from exoclaw.testing import assert_set_context_isolates_per_task


class _BuggyTool:
    """Stores destination on a plain instance attribute — the
    cross-wire bug the helper exists to catch."""

    def __init__(self) -> None:
        self.destination = ""

    def set(self, value: str) -> None:
        self.destination = value


class _CorrectTool:
    """Stores destination in a ContextVar — per-task isolation."""

    def __init__(self) -> None:
        self._var: ContextVar[str] = ContextVar(f"correct_tool_{id(self)}", default="")

    def set(self, value: str) -> None:
        self._var.set(value)

    @property
    def destination(self) -> str:
        return self._var.get()


@pytest.mark.asyncio
async def test_helper_passes_for_contextvar_backed_tool() -> None:
    """A tool that backs destination with a ContextVar must pass."""
    await assert_set_context_isolates_per_task(
        make_tool=_CorrectTool,
        set_context=lambda t, v: t.set(v),
        read_context=lambda t: t.destination,
    )


@pytest.mark.asyncio
async def test_helper_fails_for_instance_attr_backed_tool() -> None:
    """A tool that stores destination on an instance attribute must
    trip the helper's assertion — confirming the helper actually
    detects the bug it advertises."""
    with pytest.raises(AssertionError, match="ContextVar|cross-wire|race"):
        await assert_set_context_isolates_per_task(
            make_tool=_BuggyTool,
            set_context=lambda t, v: t.set(v),
            read_context=lambda t: t.destination,
        )


@pytest.mark.asyncio
async def test_helper_accepts_async_setters_and_readers() -> None:
    """``set_context`` and ``read_context`` may be async — common
    when they need to acquire a lock or do I/O. The helper's
    ``_maybe_await`` should handle either shape."""

    class _AsyncCorrectTool:
        def __init__(self) -> None:
            self._var: ContextVar[str] = ContextVar(f"async_correct_{id(self)}", default="")

        async def aset(self, value: str) -> None:
            self._var.set(value)

        async def aread(self) -> str:
            return self._var.get()

    await assert_set_context_isolates_per_task(
        make_tool=_AsyncCorrectTool,
        set_context=lambda t, v: t.aset(v),
        read_context=lambda t: t.aread(),
    )
