"""IterationPolicy protocol — pluggable termination strategy for the agent loop.

The default behaviour (no policy) is a hard ``max_iterations`` counter.
Provide an ``IterationPolicy`` to replace that with pattern-based detection,
adaptive budgets, or any other strategy — without touching the Executor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IterationPolicy(Protocol):
    """Controls when the agent loop should stop iterating."""

    async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
        """Return ``True`` to allow the next iteration, ``False`` to stop.

        *iteration* is the number of completed iterations (0 before the first).
        *tools_used* is the accumulated list of tool names called so far.
        """
        ...

    async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
        """Return the user-facing message when the loop is terminated."""
        ...
