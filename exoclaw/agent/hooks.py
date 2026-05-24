"""Agent lifecycle hooks — the generic, skill-blind contract.

The agent loop fires hooks at lifecycle seams (currently ``before_tool`` and
``before_finish``) and asks the Conversation which hooks are active for the
current turn — exactly the way it already asks ``Conversation.active_tools()``
which optional tools to advertise. Core does NOT know where those hooks come
from; in practice a skill-aware Conversation returns the hooks its active
skills registered, but **the skill concept lives entirely in the conversation
layer, not here.** Core only defines: the context handed to a hook, the
decision shapes a hook returns, and the priority-ordered dispatch.

The contract (events, decision shapes, priority-ordered merge) is ported from
openclaw's plugin hook system. Two things differ: openclaw's plugin hooks are
always-on globals, whereas here activation is whatever the Conversation reports
per turn (so a skill-aware Conversation makes them conditional); and a hook
reaches back into the runtime through an in-process ``HookContext`` rather than
a foreign script.

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

# A hook handler takes the context and returns an event-specific result (or
# ``None`` for "no opinion").
HookHandler = Callable[["HookContext"], Awaitable[Any]]


if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass

    @dataclass
    class BeforeToolResult:
        """Decision from a ``before_tool`` hook.

        ``params`` (when not ``None``) replaces the tool's arguments for the
        next hook and the eventual call — how a hook stamps an authoritative
        value. ``block=True`` refuses the call; ``block_reason`` becomes the
        tool result the model sees (a positive nudge, not an error).
        """

        params: dict[str, Any] | None = None
        block: bool = False
        block_reason: str | None = None

    @dataclass
    class BeforeFinishResult:
        """Decision from a ``before_finish`` hook.

        A non-empty ``continue_message`` is appended as a user turn and the
        loop continues (the model gets another turn); ``None``/empty lets the
        turn end.
        """

        continue_message: str | None = None

    @dataclass
    class HookContext:
        """Handle the loop passes to each hook — its door back into the runtime."""

        event: str
        run_context: dict[str, Any]
        messages: list[dict[str, Any]]
        run_effect: Callable[..., Awaitable[Any]]
        # Event-specific fields, populated by the loop per seam.
        tool_name: str | None = None
        params: dict[str, Any] | None = None
        final_content: str | None = None
        tools_used: list[str] | None = None

    @dataclass
    class HookRegistration:
        """One active hook: its handler and a priority (higher runs first).

        What ``Conversation.active_hooks(event)`` returns. Core treats it as an
        opaque (handler, priority) pair — it does not know the conversation
        derived it from a loaded skill.
        """

        handler: HookHandler
        priority: int = 0

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

    class HookRegistration:
        def __init__(self, handler: HookHandler, priority: int = 0) -> None:
            self.handler = handler
            self.priority = priority


def _ordered(regs: list[HookRegistration]) -> list[HookRegistration]:
    # Higher priority first; stable so same-priority hooks keep their order
    # (deterministic across durable replay).
    return sorted(regs, key=lambda r: -r.priority)


async def dispatch_before_tool(regs: list[HookRegistration], ctx: HookContext) -> BeforeToolResult:
    """Run ``before_tool`` hooks as ordered middleware over the tool call.

    Param mutations **compose**: each hook sees the args as mutated by the
    higher-priority hooks before it, so e.g. a cycle-id stamp and a redaction
    hook stack cleanly. The first hook to ``block`` wins and short-circuits — a
    refusal is a policy decision a lower-priority hook shouldn't override.
    """
    result = BeforeToolResult(params=ctx.params)
    for reg in _ordered(regs):
        ctx.params = result.params  # feed accumulated mutations forward
        out = await reg.handler(ctx)
        if out is None:
            continue
        if out.params is not None:
            result.params = out.params
        if out.block:
            result.block = True
            result.block_reason = out.block_reason
            break
    return result


async def dispatch_before_finish(
    regs: list[HookRegistration], ctx: HookContext
) -> BeforeFinishResult:
    """Run ``before_finish`` hooks in priority order; the highest-priority
    non-empty ``continue_message`` decides the re-prompt. One re-prompt per
    stop — the host caps repeats and ``max_iterations`` backstops."""
    for reg in _ordered(regs):
        out = await reg.handler(ctx)
        if out is not None and out.continue_message:
            return BeforeFinishResult(continue_message=out.continue_message)
    return BeforeFinishResult()


async def passthrough_effect(
    fn: Callable[..., Awaitable[object]], *args: object, **kwargs: object
) -> object:
    """Default ``HookContext.run_effect`` when the executor doesn't provide one
    (an executor predating ``run_effect``). Runs the effect inline; durable
    executors override ``run_effect`` to journal it for replay safety instead."""
    return await fn(*args, **kwargs)
