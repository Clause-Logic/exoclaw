"""Tests for exoclaw/agent/tools/registry.py coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from exoclaw.agent.tools.registry import ToolRegistry


def _make_tool(name: str = "my_tool", execute_return: str = "ok") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.cast_params = MagicMock(side_effect=lambda p: p)
    tool.validate_params = MagicMock(return_value=[])
    tool.execute = AsyncMock(return_value=execute_return)
    tool.to_schema = MagicMock(return_value={"type": "function", "function": {"name": name}})
    return tool


class TestToolRegistryBasics:
    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        tool = _make_tool("search")
        reg.register(tool)
        assert reg.get("search") is tool

    def test_get_missing_returns_none(self) -> None:
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_has_registered(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("t"))
        assert reg.has("t")

    def test_has_unregistered(self) -> None:
        reg = ToolRegistry()
        assert not reg.has("t")

    def test_unregister_existing(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("t"))
        reg.unregister("t")
        assert not reg.has("t")

    def test_unregister_nonexistent_no_error(self) -> None:
        reg = ToolRegistry()
        reg.unregister("ghost")  # Should not raise

    def test_tool_names(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        assert set(reg.tool_names) == {"a", "b"}

    def test_get_definitions(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("t"))
        defs = reg.get_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "t"


class TestToolRegistryExecute:
    async def test_execute_tool_not_found(self) -> None:
        reg = ToolRegistry()
        result = await reg.execute("missing_tool", {})
        assert "not found" in result.lower() or "Error" in result

    async def test_execute_success(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("search", execute_return="results"))
        result = await reg.execute("search", {"q": "test"})
        assert result == "results"

    async def test_execute_tool_raises_exception(self) -> None:
        reg = ToolRegistry()
        tool = _make_tool("broken")
        tool.execute = AsyncMock(side_effect=RuntimeError("something went wrong"))
        reg.register(tool)
        result = await reg.execute("broken", {})
        assert "Error" in result
        assert "something went wrong" in result

    async def test_execute_returns_error_string_appends_hint(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("errtool", execute_return="Error: something failed"))
        result = await reg.execute("errtool", {})
        assert "Error" in result
        assert "different approach" in result

    async def test_execute_validation_errors(self) -> None:
        reg = ToolRegistry()
        tool = _make_tool("validated")
        tool.validate_params = MagicMock(return_value=["missing required field"])
        reg.register(tool)
        result = await reg.execute("validated", {})
        assert "Invalid parameters" in result or "missing required" in result

    async def test_execute_params_cast_called(self) -> None:
        reg = ToolRegistry()
        tool = _make_tool("t")
        original_params = {"x": "1"}
        casted = {"x": 1}
        tool.cast_params = MagicMock(return_value=casted)
        reg.register(tool)
        await reg.execute("t", original_params)
        tool.cast_params.assert_called_once_with(original_params)
        tool.execute.assert_called_once_with(**casted)

    async def test_execute_non_error_string_no_hint(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("t", execute_return="Success: all done"))
        result = await reg.execute("t", {})
        assert result == "Success: all done"
