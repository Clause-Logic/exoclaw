"""ToolRegistry behaviour on MicroPython.

Proves ``ToolRegistry`` dispatches to a registered tool, surfaces
``not found`` errors, and binds the dispatch task-local so
``ToolRegistry.current()`` is observable from inside the tool body.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

import asyncio

from exoclaw.agent.tools.registry import ToolRegistry


class _Echo:
    """Trivial tool: returns its ``msg`` arg verbatim."""

    name = "echo"
    description = "echoes input"
    parameters = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    async def execute(self, **kw):
        return kw.get("msg", "")


def test_register_and_lookup():
    reg = ToolRegistry()
    assert reg.tool_names == []
    reg.register(_Echo())
    assert "echo" in reg.tool_names
    assert reg.has("echo") is True
    assert reg.get("echo") is not None


def test_unregister():
    reg = ToolRegistry()
    reg.register(_Echo())
    reg.unregister("echo")
    assert reg.has("echo") is False


def test_execute_dispatches():
    reg = ToolRegistry()
    reg.register(_Echo())

    async def _go():
        result = await reg.execute("echo", {"msg": "hi"})
        assert result == "hi", "expected 'hi' got {!r}".format(result)

    asyncio.run(_go())


def test_execute_unknown_tool_returns_error_string():
    """Unknown-tool results are returned as strings (agent-visible
    failures), not raised — so the agent loop can show the LLM the
    error and let it retry."""
    reg = ToolRegistry()

    async def _go():
        result = await reg.execute("missing", {})
        assert "not found" in result.lower(), "expected not-found message: {!r}".format(result)

    asyncio.run(_go())


def test_current_observable_from_tool_body():
    """``ToolRegistry.current()`` returns the dispatching registry
    while a tool body is running. This is what fan-out tools
    (``BatchTool``, ``ReduceTool``) use to look up sibling tools."""
    reg = ToolRegistry()
    seen = []

    class _Probe:
        name = "probe"
        description = "captures current registry"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            seen.append(ToolRegistry.current())
            return "ok"

    reg.register(_Probe())

    async def _go():
        await reg.execute("probe", {})
        assert seen == [reg], "current() should be the dispatching registry, got {!r}".format(seen)
        # After dispatch, current() returns to None.
        assert ToolRegistry.current() is None

    asyncio.run(_go())


def test_bind_dispatch_context_manager():
    """``bind_dispatch`` is the standalone variant for the streaming
    path. Inside the ``with`` block ``current()`` is bound; on exit
    it returns to the previous value."""
    reg = ToolRegistry()
    assert ToolRegistry.current() is None
    with reg.bind_dispatch():
        assert ToolRegistry.current() is reg
    assert ToolRegistry.current() is None
