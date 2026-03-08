"""Tool protocol — the only surface external tool packages need to satisfy."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """
    Structural protocol for agent tools.

    External packages implement this without inheriting from any nanobot class:

        class MyTool:
            name = "my_tool"
            description = "Does something useful."
            parameters = {"type": "object", "properties": {...}, "required": [...]}

            async def execute(self, **kwargs: Any) -> str:
                ...

    Use nanobot.agent.tools.base.ToolBase as an optional mixin to get
    cast_params / validate_params / to_schema utilities for free.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    async def execute(self, **kwargs: Any) -> str: ...
