"""Tests for exoclaw/agent/tools/registry.py coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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

    async def test_execute_tool_raises_exception_propagates(self) -> None:
        # Tool exceptions propagate so AgentLoop can observe them on its
        # tool span. Domain errors (not-found, invalid params) still return
        # strings because they're normal agent-visible outcomes.
        reg = ToolRegistry()
        tool = _make_tool("broken")
        tool.execute = AsyncMock(side_effect=RuntimeError("something went wrong"))
        reg.register(tool)
        with pytest.raises(RuntimeError, match="something went wrong"):
            await reg.execute("broken", {})

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


class _PlainTool:
    """Minimal tool that doesn't have ``cast_params`` / ``validate_params``.

    ``_make_tool`` uses ``MagicMock`` which auto-creates every attribute,
    so ``hasattr(tool, "validate_params")`` is always True and the
    registry's validation path returns a MagicMock that truthy-checks as
    an error — swallowing the call before the tool body runs. Use this
    plain class for tests that need to observe side effects inside the
    tool body.
    """

    def __init__(self, name: str, body: "AsyncMock | object") -> None:
        self.name = name
        self._body = body

    async def execute(self, **kwargs: object) -> str:
        result = self._body()
        if hasattr(result, "__await__"):
            return await result  # type: ignore[no-any-return]
        return result  # type: ignore[return-value]


class TestToolRegistryDispatchContextVar:
    """``ToolRegistry.execute`` binds ``self`` into a ContextVar for the
    duration of the tool body so fan-out tools can look up sibling
    tools via :meth:`ToolRegistry.current` without a stored reference.

    The stored-reference pattern (``set_registry``) breaks when a
    single tool instance is shared across multiple ``AgentLoop``s —
    each loop's constructor overwrites the pointer. The ContextVar
    is per-asyncio-task, so concurrent dispatches from different
    loops never clobber each other. These tests pin the invariants
    that fix is relying on.
    """

    async def test_current_is_none_outside_dispatch(self) -> None:
        assert ToolRegistry.current() is None

    async def test_current_returns_dispatching_registry(self) -> None:
        """The tool body must see ``current()`` return the registry
        that's dispatching it — that's the whole point.
        """
        captured: dict[str, "ToolRegistry | None"] = {}
        reg = ToolRegistry()

        async def _body() -> str:
            captured["current"] = ToolRegistry.current()
            return "ok"

        reg.register(_PlainTool("t", _body))  # type: ignore[arg-type]
        await reg.execute("t", {})
        assert captured["current"] is reg

    async def test_current_restored_to_none_after_execute(self) -> None:
        reg = ToolRegistry()

        async def _body() -> str:
            return "ok"

        reg.register(_PlainTool("t", _body))  # type: ignore[arg-type]
        await reg.execute("t", {})
        assert ToolRegistry.current() is None

    async def test_current_restored_on_tool_exception(self) -> None:
        reg = ToolRegistry()

        async def _boom() -> str:
            raise RuntimeError("nope")

        reg.register(_PlainTool("boom", _boom))  # type: ignore[arg-type]

        with pytest.raises(RuntimeError):
            await reg.execute("boom", {})
        assert ToolRegistry.current() is None, (
            "dispatch ContextVar must be reset in a finally block — "
            "an exception in the tool body should not leak stale state"
        )

    async def test_nested_execute_restores_outer_on_return(self) -> None:
        """A fan-out tool that dispatches back into the registry (e.g.
        ``BatchTool``) must not corrupt the outer dispatch binding.
        After the inner call returns, ``current()`` has to point at
        the outer registry again.
        """
        outer = ToolRegistry()
        inner = ToolRegistry()
        seen: list["ToolRegistry | None"] = []

        async def _inner_body() -> str:
            seen.append(ToolRegistry.current())
            return "inner"

        inner.register(_PlainTool("inner", _inner_body))  # type: ignore[arg-type]

        async def _outer_body() -> str:
            seen.append(ToolRegistry.current())
            await inner.execute("inner", {})
            seen.append(ToolRegistry.current())
            return "outer"

        outer.register(_PlainTool("outer", _outer_body))  # type: ignore[arg-type]

        await outer.execute("outer", {})

        assert seen == [outer, inner, outer], f"nested dispatch mishandled: {seen}"
        assert ToolRegistry.current() is None

    async def test_shared_tool_across_two_registries(self) -> None:
        """The exact production bug we're fixing.

        One tool instance (e.g. a ``BatchTool``) is registered in two
        separate ``ToolRegistry`` objects — one for a main agent, one
        for a subagent. When each registry dispatches that tool, the
        tool body must observe the *dispatching* registry via
        ``current()``, not whichever registry happened to be stored
        last on the shared instance.

        This test fails on the pre-fix code because the only way to
        know which registry is active is a stored reference on the
        tool, which is last-write-wins.
        """
        main_reg = ToolRegistry()
        sub_reg = ToolRegistry()

        observed: list[str] = []

        async def _tool_body() -> str:
            current = ToolRegistry.current()
            assert current is not None
            observed.append("main" if current is main_reg else "sub")
            return "ok"

        shared_tool = _PlainTool("shared", _tool_body)
        main_reg.register(shared_tool)  # type: ignore[arg-type]
        sub_reg.register(shared_tool)  # type: ignore[arg-type]

        await main_reg.execute("shared", {})
        await sub_reg.execute("shared", {})
        await main_reg.execute("shared", {})

        assert observed == ["main", "sub", "main"], (
            f"shared tool saw wrong registry sequence: {observed}"
        )
