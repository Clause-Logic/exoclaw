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


def test_execute_with_context_path():
    """Tools that implement ``execute_with_context(ctx, **kwargs)``
    take precedence over plain ``execute`` when the registry is
    called with a non-None ctx. Same protocol used by tools that
    need session routing."""
    from exoclaw.agent.tools.protocol import ToolContext

    seen_ctx = []

    class _CtxAware:
        name = "ctxa"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "plain"

        async def execute_with_context(self, ctx, **kw):
            seen_ctx.append(ctx)
            return "ctx-aware"

    reg = ToolRegistry()
    reg.register(_CtxAware())

    async def _go():
        ctx = ToolContext(session_key="s", channel="c", chat_id="i")
        result = await reg.execute("ctxa", {}, ctx=ctx)
        assert result == "ctx-aware"
        assert seen_ctx == [ctx]

    asyncio.run(_go())


def test_execute_error_result_gets_hint_appended():
    """When a tool returns a string starting with ``Error``, the
    registry appends a ``[Analyze the error above...]`` hint so
    the LLM sees actionable guidance. Mirrors how the registry
    standardizes error UX."""

    class _Failing:
        name = "fail"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "Error: something broke"

    reg = ToolRegistry()
    reg.register(_Failing())

    async def _go():
        result = await reg.execute("fail", {})
        assert result.startswith("Error: something broke")
        assert "Analyze the error above" in result

    asyncio.run(_go())


def test_resolve_with_invalid_params_returns_error():
    """A tool with ``validate_params`` rejecting input → registry
    short-circuits with an ``Invalid parameters for tool`` string."""
    from exoclaw.agent.tools.protocol import ToolBase

    class _Strict(ToolBase):
        name = "strict"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }

        async def execute(self, **kw):
            return "should-not-run"

    reg = ToolRegistry()
    reg.register(_Strict())

    async def _go():
        # Missing required ``n`` → validation error path.
        result = await reg.execute("strict", {})
        assert result.startswith("Error: Invalid parameters")
        assert "missing" in result.lower()

    asyncio.run(_go())


def test_resolve_casts_params_through_cast_params():
    """``cast_params`` runs before validation — strings get coerced
    to the schema's declared type, then validation passes."""
    from exoclaw.agent.tools.protocol import ToolBase

    seen = []

    class _CastedInt(ToolBase):
        name = "ci"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }

        async def execute(self, **kw):
            seen.append(kw)
            return "ok"

    reg = ToolRegistry()
    reg.register(_CastedInt())

    async def _go():
        # Pass a string; cast_params coerces it before execute.
        result = await reg.execute("ci", {"n": "42"})
        assert result == "ok"
        assert seen == [{"n": 42}]

    asyncio.run(_go())


def test_stream_dispatch_returns_tool_and_params():
    """``stream_dispatch`` is the no-execute resolution path the
    streaming executor uses: it returns ``(tool, validated_params)``
    or an error string."""

    class _Plain:
        name = "p"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

    reg = ToolRegistry()
    reg.register(_Plain())
    out = reg.stream_dispatch("p", {})
    assert isinstance(out, tuple)
    tool, params = out
    assert tool.name == "p"
    assert params == {}


def test_stream_dispatch_unknown_tool_returns_error():
    reg = ToolRegistry()
    out = reg.stream_dispatch("missing", {})
    assert isinstance(out, str)
    assert "not found" in out.lower()


def test_bind_dispatch_context_manager():
    """``bind_dispatch`` is the standalone variant for the streaming
    path. Inside the ``with`` block ``current()`` is bound; on exit
    it returns to the previous value."""
    reg = ToolRegistry()
    assert ToolRegistry.current() is None
    with reg.bind_dispatch():
        assert ToolRegistry.current() is reg
    assert ToolRegistry.current() is None
