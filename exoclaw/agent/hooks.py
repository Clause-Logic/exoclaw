"""Agent lifecycle hooks — the generic contract.

The agent loop fires hooks at lifecycle seams (currently ``before_tool`` and
``before_finish``). At each seam it builds a ``HookContext`` and asks the
Conversation for a single decision — the same way it already asks
``Conversation.active_tools()`` which optional tools to advertise. Core defines
only three things: the **context** handed to a hook, the **decision shapes** a
hook returns, and the seams the loop calls. It has no opinion on what produces
those decisions, how many hooks there are, or how they compose — a consumer
decides activation and collapses any number of hooks into the one result core
applies.

The decision shapes (mutate args / veto a tool, re-prompt a stopped model) are
ported from openclaw's plugin hook system. Two things differ: activation is
per-turn (whatever the Conversation reports), not always-on globals; and a hook
reaches back into the runtime through an in-process ``HookContext`` rather than
a foreign script. (The priority-ordered *runner* that merges multiple hooks
into one decision lives in the consumer, exactly as openclaw keeps its hook
runner in the plugin layer rather than the agent core.)

``HookContext`` is the hook's only door back into the runtime. Keeping I/O
behind ``run_effect`` means a hook author writes plain async code and can't
break durable replay — the executor owns journaling. ``run_context`` is a
per-run bag the host seeds (e.g. a cycle id) so a hook can read authoritative
values instead of trusting the model's tool args; ``messages`` is the current
transcript (read-only) so e.g. a budget hook counts prior calls instead of
holding a counter that wouldn't survive replay.

Dual-class pattern (see ``exoclaw/bus/events.py``): MicroPython strips
``name: type`` annotations at compile time, so a runtime ``@dataclass`` can't
build ``__init__``. CPython gets real dataclasses; MP gets hand-written
``__init__`` — only the construction machinery differs.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from exoclaw._compat import IS_MICROPYTHON

# Event names. Plain strings (not an enum) so callers register against them
# without importing a symbol — mirrors how openclaw keys hooks by name.
BEFORE_TOOL = "before_tool"
BEFORE_FINISH = "before_finish"


if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass

    @dataclass
    class BeforeToolResult:
        """The decision a ``before_tool`` consumer returns.

        ``params`` (when not ``None``) replaces the tool's arguments — how a
        consumer stamps an authoritative value. ``block=True`` refuses the
        call; ``block_reason`` becomes the tool result the model sees (a
        positive nudge, not an error).
        """

        params: dict[str, Any] | None = None
        block: bool = False
        block_reason: str | None = None

    @dataclass
    class BeforeFinishResult:
        """The decision a ``before_finish`` consumer returns.

        A non-empty ``continue_message`` is appended as a user turn and the
        loop continues (the model gets another turn); ``None``/empty lets the
        turn end.
        """

        continue_message: str | None = None

    @dataclass
    class HookContext:
        """Handle the loop passes to a hook decision — its door back into the
        runtime."""

        event: str
        run_context: dict[str, Any]
        messages: list[dict[str, Any]]
        run_effect: Callable[..., Awaitable[Any]]
        # Event-specific fields, populated by the loop per seam.
        tool_name: str | None = None
        params: dict[str, Any] | None = None
        final_content: str | None = None
        tools_used: list[str] | None = None

else:  # pragma: no cover (cpython)

    class BeforeToolResult:
        def __init__(
            self,
            params: dict[str, Any] | None = None,
            block: bool = False,
            block_reason: str | None = None,
        ) -> None:
            self.params = params
            self.block = block
            self.block_reason = block_reason

    class BeforeFinishResult:
        def __init__(self, continue_message: str | None = None) -> None:
            self.continue_message = continue_message

    class HookContext:
        def __init__(
            self,
            event: str,
            run_context: dict[str, Any],
            messages: list[dict[str, Any]],
            run_effect: Callable[..., Awaitable[Any]],
            tool_name: str | None = None,
            params: dict[str, Any] | None = None,
            final_content: str | None = None,
            tools_used: list[str] | None = None,
        ) -> None:
            self.event = event
            self.run_context = run_context
            self.messages = messages
            self.run_effect = run_effect
            self.tool_name = tool_name
            self.params = params
            self.final_content = final_content
            self.tools_used = tools_used


async def passthrough_effect(
    fn: Callable[..., Awaitable[object]], *args: object, **kwargs: object
) -> object:
    """Default ``HookContext.run_effect`` when the executor doesn't provide one
    (an executor predating ``run_effect``). Runs the effect inline; durable
    executors override ``run_effect`` to journal it for replay safety instead."""
    return await fn(*args, **kwargs)
