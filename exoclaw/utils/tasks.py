"""Helpers for spawning asyncio tasks safely.

Background ``asyncio.create_task`` calls copy the parent task's
``contextvars`` snapshot. When the parent is inside a DBOS workflow
or step, that snapshot includes ``DBOSContext`` (workflow_id +
function_id counter). The spawned task then carries a stale or live
DBOSContext that can mis-classify any subsequent
``DBOS.start_workflow_async`` / step call.

Use ``create_isolated_task`` instead of ``asyncio.create_task`` for
any background work that should run independently of the caller's
DBOS / structlog / per-turn contextvars.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

from exoclaw._compat import IS_MICROPYTHON

_T = TypeVar("_T")


def create_isolated_task(
    coro: Coroutine[Any, Any, _T],
    *,
    name: str | None = None,
) -> asyncio.Task[_T]:
    """Spawn an asyncio task with a fresh ``contextvars.Context``.

    The new task does not inherit the caller's contextvar bindings —
    most importantly DBOS's ``_dbos_context_var``, but also any
    structlog ``bind_contextvars`` state and other per-turn vars.

    Use this for any background task that should run as a top-level
    unit of work (timer callbacks, background subagents, deferred
    consolidation, etc.).

    On MicroPython, ``uasyncio.create_task`` doesn't accept
    ``context`` or ``name`` kwargs. We wrap the user's coroutine in
    a ``finally`` that drops the task's per-task storage entry from
    ``_compat._task_storage`` — that's how the
    ``id(asyncio.current_task())``-keyed ``TaskLocal`` storage gets
    cleaned up. Without this wrapper a long-running process would
    accumulate a stale dict per finished task.

    Inheritance from the parent's contextvar bindings is NOT
    something the MP path defends against: there's no contextvars
    layer to leak, and the per-task storage is freshly empty for
    the new task.
    """
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        from exoclaw._compat import _drop_current_task_storage

        async def _wrapped() -> _T:
            try:
                return await coro
            finally:
                _drop_current_task_storage()

        return asyncio.create_task(_wrapped())
    import contextvars  # pragma: no cover (micropython)

    return asyncio.create_task(  # pragma: no cover (micropython)
        coro, context=contextvars.Context(), name=name
    )


__all__ = ["create_isolated_task"]
