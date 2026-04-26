"""Tool registry for dynamic tool management."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

from exoclaw.agent.tools.protocol import Tool, ToolContext

# ContextVar that carries the *currently-dispatching* ToolRegistry across
# ``ToolRegistry.execute`` and whatever tool body it invokes. Tools that
# need to look up sibling tools (``BatchTool``, ``ReduceTool``,
# fan-out/fan-in patterns) should read this via
# :meth:`ToolRegistry.current` instead of holding a stored reference via
# ``set_registry()``: the stored-reference pattern breaks when a single
# tool instance is shared across multiple ``AgentLoop``s — each loop's
# constructor overwrites the pointer, so the main agent's tool ends up
# pointing at the last-constructed subagent's registry. Per-asyncio-task
# ContextVar isolation fixes that because concurrent dispatches from
# different loops each see their own registry.
_dispatch_registry: contextvars.ContextVar["ToolRegistry | None"] = contextvars.ContextVar(
    "exoclaw.tool_registry.dispatch", default=None
)


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.

    Tools may be registered as *optional* — they are loaded and executable
    but hidden from the LLM's tool list by default.  Pass their names via
    ``include`` in :meth:`get_definitions` to surface them for a turn.
    This lets callers (e.g. a skill system) selectively activate tools
    without the registry knowing anything about skills.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._optional: set[str] = set()

    def register(self, tool: Tool, optional: bool = False) -> None:
        """Register a tool.

        Args:
            tool: The tool to register.
            optional: If True the tool is hidden from :meth:`get_definitions`
                unless its name appears in the ``include`` argument.
        """
        self._tools[tool.name] = tool
        if optional:
            self._optional.add(tool.name)

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._optional.discard(name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self, include: "set[str] | None" = None) -> list[dict[str, object]]:
        """Get tool definitions in OpenAI format.

        Args:
            include: Optional set of optional-tool names to surface.
                Optional tools whose names are *not* in this set are omitted.
                Non-optional tools are always included.
                Pass ``None`` (default) to include only non-optional tools —
                preserving backwards-compatible behaviour for callers that
                never register optional tools.
        """
        visible = [
            t
            for name, t in self._tools.items()
            if name not in self._optional or (include is not None and name in include)
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in visible
        ]

    async def execute(
        self, name: str, params: dict[str, object], ctx: ToolContext | None = None
    ) -> str:
        """Execute a tool by name with given parameters.

        If the tool implements execute_with_context(ctx, **kwargs), that is called
        instead of execute(**kwargs). Falls back to execute() if not implemented or
        ctx is None.

        Exceptions raised by the tool propagate to the caller so the agent
        loop can observe them as part of its tool span. Domain errors (tool
        not found, invalid parameters) are still returned as strings because
        they're normal agent-visible outcomes, not unexpected failures.

        Binds ``self`` into the ``_dispatch_registry`` ContextVar for the
        duration of the tool body so fan-out tools can look up sibling
        tools via :meth:`ToolRegistry.current`. The binding is restored
        on exit — nested dispatches (a tool that calls back into the
        registry) compose correctly because each execute() frame saves
        and restores the previous value.
        """
        _hint = "\n\n[Analyze the error above and try a different approach.]"

        resolved = self._resolve(name, params)
        if isinstance(resolved, str):
            return resolved
        tool, params = resolved

        token = _dispatch_registry.set(self)
        try:
            if ctx is not None and hasattr(tool, "execute_with_context"):
                result: str = await getattr(tool, "execute_with_context")(ctx, **params)
            else:
                result = await tool.execute(**params)
        finally:
            _dispatch_registry.reset(token)

        if isinstance(result, str) and result.startswith("Error"):
            return result + _hint
        return result

    def stream_dispatch(
        self, name: str, params: dict[str, object]
    ) -> tuple[Tool, dict[str, object]] | str:
        """Resolve a tool call for a potential streaming path without
        invoking it.

        Returns the same shape as :meth:`_resolve`:
        ``(tool, validated_params)`` when the tool exists and the
        parameters are valid, or an error string for unknown tools /
        invalid params. This helper does **not** check whether the
        resolved tool implements ``execute_streaming`` — callers
        (typically ``DirectExecutor.execute_tool_with_handle``)
        inspect the returned tool themselves before starting any
        async iteration, and use :meth:`bind_dispatch` to mirror the
        ``_dispatch_registry`` ContextVar binding that
        :meth:`execute` would have provided.

        Kept separate from :meth:`execute` because the streaming path
        is invoked from the executor, not the registry — the executor
        owns the scratch file lifecycle and needs the iterator to
        consume chunks one at a time. Doing the streaming dispatch
        inside the registry would force the registry to know about
        scratch files, which is the executor's concern.
        """
        return self._resolve(name, params)

    @contextlib.contextmanager
    def bind_dispatch(self) -> Iterator[None]:
        """Bind ``self`` into ``_dispatch_registry`` for the body of
        the context.

        The streaming executor path bypasses :meth:`execute`, which
        normally sets ``_dispatch_registry`` for the duration of the
        tool body. Without this binding, fan-out tools that call
        :meth:`ToolRegistry.current` from inside their
        ``execute_streaming`` body would see ``None`` and break
        (``BatchTool``, ``ReduceTool``, anything that looks up
        sibling tools by name). The token is restored on exit so
        nested dispatches compose correctly — same semantics as
        :meth:`execute`'s set/reset pair.
        """
        token = _dispatch_registry.set(self)
        try:
            yield
        finally:
            _dispatch_registry.reset(token)

    def _resolve(
        self, name: str, params: dict[str, object]
    ) -> tuple[Tool, dict[str, object]] | str:
        """Shared lookup + cast + validate for both execute paths.

        Returns ``(tool, params)`` on success or a ready-to-return
        error string on failure. Hint suffix is applied in
        ``execute``; the streaming caller appends its own hint or
        returns the error inline.
        """
        _hint = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        if hasattr(tool, "cast_params"):
            params = getattr(tool, "cast_params")(params)
        if hasattr(tool, "validate_params"):
            errors: list[str] = getattr(tool, "validate_params")(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _hint

        return tool, params

    @classmethod
    def current(cls) -> "ToolRegistry | None":
        """Return the registry currently dispatching a tool on this task.

        Set automatically by :meth:`execute` for the duration of the
        tool body. Fan-out tools like ``BatchTool`` and ``ReduceTool``
        should prefer this over a stored reference from ``set_registry``
        because a single tool instance can be wired into multiple
        ``AgentLoop``s (each with its own registry) — the stored
        reference is last-write-wins, but the ContextVar is
        per-asyncio-task so concurrent dispatches never clobber each
        other.

        Returns ``None`` when no dispatch is in progress (e.g. the
        caller invokes a tool body directly, bypassing the registry).
        """
        return _dispatch_registry.get()

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
