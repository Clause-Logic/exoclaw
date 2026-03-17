"""Tool registry for dynamic tool management."""

from exoclaw.agent.tools.protocol import Tool, ToolContext


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
        """
        _hint = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            if hasattr(tool, "cast_params"):
                params = getattr(tool, "cast_params")(params)
            if hasattr(tool, "validate_params"):
                errors: list[str] = getattr(tool, "validate_params")(params)
                if errors:
                    return (
                        f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _hint
                    )
            if ctx is not None and hasattr(tool, "execute_with_context"):
                result: str = await getattr(tool, "execute_with_context")(ctx, **params)
            else:
                result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _hint
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _hint

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
