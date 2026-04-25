"""Behavioral coverage for ``exoclaw.utils.create_isolated_task``."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import pytest

from exoclaw.utils import create_isolated_task

_PROBE: ContextVar[str] = ContextVar("isolation_test_probe", default="default")


@pytest.mark.asyncio
async def test_isolated_task_does_not_inherit_caller_contextvar() -> None:
    """A ContextVar bound in the caller must not be visible to the
    spawned task — that's the whole point of the wrapper."""
    _PROBE.set("caller-binding")
    captured: dict[str, str] = {}

    async def child() -> None:
        captured["seen"] = _PROBE.get()

    await create_isolated_task(child())

    assert captured["seen"] == "default", (
        f"isolated task observed caller's ContextVar value "
        f"{captured['seen']!r}; expected 'default'. Wrapper failed "
        f"to start the task with a fresh contextvars.Context()."
    )


@pytest.mark.asyncio
async def test_isolated_task_returns_value_from_coroutine() -> None:
    """The returned Task should resolve to whatever the coroutine
    returns — same shape as ``asyncio.create_task``."""

    async def child() -> int:
        return 42

    task = create_isolated_task(child())
    assert await task == 42


@pytest.mark.asyncio
async def test_isolated_task_propagates_name() -> None:
    """``name=`` should reach the underlying ``asyncio.Task``."""

    async def child() -> None:
        return None

    task = create_isolated_task(child(), name="probe-task")
    assert task.get_name() == "probe-task"
    await task


@pytest.mark.asyncio
async def test_plain_create_task_does_inherit_caller_contextvar() -> None:
    """Negative control: bare ``asyncio.create_task`` DOES leak the
    caller's binding. Documents the behavior the wrapper exists to
    avoid; if this ever fails the wrapper has become a no-op."""
    _PROBE.set("caller-binding")
    captured: dict[str, str] = {}

    async def child() -> None:
        captured["seen"] = _PROBE.get()

    # Intentional bare create_task — TID251 is silenced for this file
    # via the per-file-ignore on tests/.
    await asyncio.create_task(child())

    assert captured["seen"] == "caller-binding"
