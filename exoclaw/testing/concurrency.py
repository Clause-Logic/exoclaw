"""Concurrency / contextvar isolation assertions for tool authors.

A class of bugs in tools where per-call routing state (channel,
chat_id, session_key, parent skills) is stored on plain instance
attributes mutated by ``set_context``. The instance is shared
across every concurrent turn, so whichever task last wrote the
attrs owns routing for every other task until someone else stomps
them — one caller's outputs end up routed to another caller's
destination.

The fix is to back per-call state with ``contextvars.ContextVar``
so each task sees its own binding. Hand-checking each new tool is
unreliable; this helper exists so any tool's test suite can call
``assert_set_context_isolates_per_task(...)`` and get a
deterministic regression check.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar, cast

from exoclaw._compat import isawaitable

_TTool = TypeVar("_TTool")
_TVal = TypeVar("_TVal")


async def _maybe_await(value: Awaitable[_TVal] | _TVal) -> _TVal:
    """Await value if it's awaitable, otherwise return as-is.

    Uses ``inspect.isawaitable`` so Future / Task / custom awaitables
    are handled, not just bare coroutines."""
    if isawaitable(value):
        return await cast("Awaitable[_TVal]", value)
    return cast(_TVal, value)


async def assert_set_context_isolates_per_task(
    *,
    make_tool: Callable[[], _TTool],
    set_context: Callable[[_TTool, str], Awaitable[None] | None],
    read_context: Callable[[_TTool], Awaitable[str] | str],
) -> None:
    """Assert that a tool's ``set_context`` binding is per-task.

    Spawns two concurrent tasks that each ``set_context`` to a
    different value, both yield to let the other interleave, then
    each ``read_context``. Each task must observe its own value.
    With instance-attr storage, the second writer's value is
    visible to both readers (the singleton bug). With ContextVar
    storage, each task sees its own.

    Parameters
    ----------
    make_tool:
        Zero-arg factory returning a fresh tool instance.
    set_context:
        Coroutine or sync function that binds the tool's
        per-call destination to a string. Typically a thin wrapper
        over the tool's ``set_context`` method.
    read_context:
        Coroutine or sync function returning the tool's current
        per-task destination as a string. Typically a property
        accessor or a re-read of the tool's destination attribute.

    Example
    -------
    ::

        await assert_set_context_isolates_per_task(
            make_tool=lambda: MessageTool(send_callback=AsyncMock()),
            set_context=lambda t, v: t.set_context(channel="zulip", chat_id=v),
            read_context=lambda t: t._default_chat_id,
        )
    """
    tool = make_tool()
    observed: dict[str, str] = {}

    async def turn(tag: str) -> None:
        await _maybe_await(set_context(tool, tag))
        # Yield so the other task's set_context can interleave —
        # this is the moment the singleton bug would cross-wire.
        # ``sleep(0)`` is a deterministic zero-duration yield; a fixed
        # delay would just slow the helper without making the race
        # any more reliable.
        await asyncio.sleep(0)
        observed[tag] = await _maybe_await(read_context(tool))

    await asyncio.gather(turn("ALPHA"), turn("BRAVO"))

    assert observed.get("ALPHA") == "ALPHA", (
        f"ALPHA task observed {observed.get('ALPHA')!r} after BRAVO's "
        f"set_context — destination is shared instance state, not a "
        f"per-task ContextVar. Concurrent turns will cross-wire each "
        f"other's routing."
    )
    assert observed.get("BRAVO") == "BRAVO", (
        f"BRAVO task observed {observed.get('BRAVO')!r} after ALPHA's "
        f"set_context — same race in the opposite direction."
    )


__all__ = ["assert_set_context_isolates_per_task"]
