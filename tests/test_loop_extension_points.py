"""Tests for AgentLoop extension points: callbacks, ToolContext, execute_with_context, set_bus."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.queue import MessageBus


def _make_response(
    content: str | None = "hello",
    has_tool_calls: bool = False,
    finish_reason: str = "stop",
) -> MagicMock:
    r = MagicMock()
    r.has_tool_calls = has_tool_calls
    r.content = content
    r.finish_reason = finish_reason
    r.tool_calls = []
    r.reasoning_content = None
    r.thinking_blocks = None
    return r


def _make_tool_call(
    name: str = "my_tool",
    args: dict[str, object] | None = None,
    call_id: str = "tc1",
) -> MagicMock:
    tc = MagicMock()
    tc.name = name
    tc.arguments = args or {}
    tc.id = call_id
    return tc


def _make_loop(**kwargs: object) -> tuple[AgentLoop, MessageBus]:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(return_value=_make_response())
    conversation = MagicMock()
    conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    conversation.record = AsyncMock()
    conversation.clear = AsyncMock(return_value=True)
    loop = AgentLoop(bus=bus, provider=provider, conversation=conversation, **kwargs)
    return loop, bus


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------


class TestToolContext:
    def test_fields(self) -> None:
        ctx = ToolContext(session_key="cli:main", channel="cli", chat_id="main")
        assert ctx.session_key == "cli:main"
        assert ctx.channel == "cli"
        assert ctx.chat_id == "main"


# ---------------------------------------------------------------------------
# execute_with_context in ToolRegistry
# ---------------------------------------------------------------------------


class TestExecuteWithContext:
    async def test_calls_execute_with_context_when_present(self) -> None:
        reg = ToolRegistry()
        tool = MagicMock()
        tool.name = "ctx_tool"
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="plain")
        tool.execute_with_context = AsyncMock(return_value="context-aware result")
        reg.register(tool)

        ctx = ToolContext(session_key="s", channel="cli", chat_id="main")
        result = await reg.execute("ctx_tool", {}, ctx=ctx)
        assert result == "context-aware result"
        tool.execute_with_context.assert_called_once_with(ctx)
        tool.execute.assert_not_called()

    async def test_falls_back_to_execute_when_no_ctx(self) -> None:
        reg = ToolRegistry()
        tool = MagicMock()
        tool.name = "ctx_tool"
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="plain")
        tool.execute_with_context = AsyncMock(return_value="context-aware result")
        reg.register(tool)

        result = await reg.execute("ctx_tool", {}, ctx=None)
        assert result == "plain"
        tool.execute_with_context.assert_not_called()

    async def test_falls_back_when_tool_has_no_execute_with_context(self) -> None:
        reg = ToolRegistry()
        tool = MagicMock(
            spec=["name", "description", "parameters", "execute", "cast_params", "validate_params"]
        )
        tool.name = "simple_tool"
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="simple result")
        reg.register(tool)

        ctx = ToolContext(session_key="s", channel="cli", chat_id="main")
        result = await reg.execute("simple_tool", {}, ctx=ctx)
        assert result == "simple result"


# ---------------------------------------------------------------------------
# set_bus — duck-typed hook called at registration
# ---------------------------------------------------------------------------


class TestSetBus:
    def test_set_bus_called_on_tool_registration(self) -> None:
        tool = MagicMock()
        tool.name = "bus_tool"
        tool.set_bus = MagicMock()

        loop, bus = _make_loop(tools=[tool])
        tool.set_bus.assert_called_once_with(bus)

    def test_tools_without_set_bus_are_unaffected(self) -> None:
        tool = MagicMock(spec=["name", "description", "parameters", "execute"])
        tool.name = "plain_tool"
        # Should not raise even though set_bus is not implemented
        loop, _ = _make_loop(tools=[tool])


# ---------------------------------------------------------------------------
# set_registry — duck-typed hook called at registration
# ---------------------------------------------------------------------------


class TestSetRegistry:
    def test_set_registry_called_on_tool_registration(self) -> None:
        tool = MagicMock()
        tool.name = "registry_tool"
        tool.set_registry = MagicMock()

        loop, _ = _make_loop(tools=[tool])
        tool.set_registry.assert_called_once_with(loop.tools)

    def test_tools_without_set_registry_are_unaffected(self) -> None:
        tool = MagicMock(spec=["name", "description", "parameters", "execute"])
        tool.name = "plain_tool"
        # Should not raise even though set_registry is not implemented
        loop, _ = _make_loop(tools=[tool])


# ---------------------------------------------------------------------------
# Injectable registry
# ---------------------------------------------------------------------------


class TestInjectableRegistry:
    def test_injected_registry_is_used(self) -> None:
        reg = ToolRegistry()
        loop, _ = _make_loop(registry=reg)
        assert loop.tools is reg

    def test_default_registry_created_when_not_injected(self) -> None:
        loop, _ = _make_loop()
        assert isinstance(loop.tools, ToolRegistry)


# ---------------------------------------------------------------------------
# on_pre_context callback
# ---------------------------------------------------------------------------


class TestOnPreContext:
    async def test_on_pre_context_result_appended_to_plugin_context(self) -> None:
        extra_ctx = "## Extra\nsome injected context"
        on_pre_context = AsyncMock(return_value=extra_ctx)

        loop, bus = _make_loop(on_pre_context=on_pre_context)

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hello")
        await loop._process_message(msg)

        on_pre_context.assert_called_once()
        args = on_pre_context.call_args[0]
        assert args[0] == "hello"  # message
        assert args[2] == "cli"  # channel

    async def test_empty_pre_context_not_appended(self) -> None:
        on_pre_context = AsyncMock(return_value="")
        loop, bus = _make_loop(on_pre_context=on_pre_context)

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hello")
        await loop._process_message(msg)

        # build_prompt should be called — context just won't have extra
        loop.conversation.build_prompt.assert_called_once()


# ---------------------------------------------------------------------------
# on_pre_tool callback
# ---------------------------------------------------------------------------


class TestOnPreTool:
    async def test_on_pre_tool_rejection_used_as_result(self) -> None:
        tool_response = _make_response(has_tool_calls=True)
        tool_response.tool_calls = [_make_tool_call("my_tool")]
        final_response = _make_response(content="done")

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=[tool_response, final_response])

        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "A tool"
        tool.parameters = {"type": "object", "properties": {}}
        tool.execute = AsyncMock(return_value="real result")

        rejection_msg = "Rejected: not allowed"
        on_pre_tool = AsyncMock(return_value=rejection_msg)

        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            tools=[tool],
            on_pre_tool=on_pre_tool,
        )

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="go")
        await loop._process_message(msg)

        on_pre_tool.assert_called_once_with("my_tool", {}, "cli:main")
        tool.execute.assert_not_called()

    async def test_on_pre_tool_pass_through_when_no_rejection(self) -> None:
        tool_response = _make_response(has_tool_calls=True)
        tool_response.tool_calls = [_make_tool_call("my_tool")]
        final_response = _make_response(content="done")

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=[tool_response, final_response])

        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        # spec excludes execute_with_context so registry falls back to execute()
        tool = MagicMock(
            spec=["name", "description", "parameters", "execute", "cast_params", "validate_params"]
        )
        tool.name = "my_tool"
        tool.description = "A tool"
        tool.parameters = {"type": "object", "properties": {}}
        tool.cast_params = MagicMock(side_effect=lambda p: p)
        tool.validate_params = MagicMock(return_value=[])
        tool.execute = AsyncMock(return_value="real result")

        on_pre_tool = AsyncMock(return_value=None)  # no rejection

        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            tools=[tool],
            on_pre_tool=on_pre_tool,
        )

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="go")
        await loop._process_message(msg)

        tool.execute.assert_called_once()


# ---------------------------------------------------------------------------
# on_post_turn callback
# ---------------------------------------------------------------------------


class TestOnPostTurn:
    async def test_on_post_turn_fired_after_turn(self) -> None:
        on_post_turn = AsyncMock()
        loop, bus = _make_loop(on_post_turn=on_post_turn)

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hello")
        await loop._process_message(msg)

        # ensure_future schedules it — drain event loop
        await asyncio.sleep(0)
        on_post_turn.assert_called_once()
        args = on_post_turn.call_args[0]
        assert args[1] == "cli:main"  # session_key
        assert args[2] == "cli"  # channel
        assert args[3] == "main"  # chat_id


# ---------------------------------------------------------------------------
# on_max_iterations callback
# ---------------------------------------------------------------------------


class TestOnMaxIterations:
    async def test_on_max_iterations_fired_when_limit_reached(self) -> None:
        # Always return tool calls so the loop hits max_iterations
        tool_response = _make_response(has_tool_calls=True)
        tc = _make_tool_call("looping_tool")
        tool_response.tool_calls = [tc]

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(return_value=tool_response)

        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        tool = MagicMock()
        tool.name = "looping_tool"
        tool.description = "loops forever"
        tool.parameters = {"type": "object", "properties": {}}
        tool.execute = AsyncMock(return_value="still going")

        fired = asyncio.Event()

        async def _on_max(session_key: str, channel: str, chat_id: str) -> None:
            fired.set()

        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            tools=[tool],
            max_iterations=2,
            on_max_iterations=_on_max,
        )

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="go")
        await loop._process_message(msg)
        await asyncio.sleep(0)

        assert fired.is_set(), "on_max_iterations was not called when iteration limit was reached"


# ---------------------------------------------------------------------------
# Executor.should_continue / on_limit_reached (optional duck-typed methods)
# ---------------------------------------------------------------------------


class TestExecutorShouldContinue:
    async def test_executor_should_continue_controls_loop(self) -> None:
        """When the executor has should_continue(), it replaces the max_iterations check."""
        from exoclaw.executor import DirectExecutor

        class PolicyExecutor(DirectExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
                self.call_count += 1
                return iteration < 3  # allow 3 iterations regardless of max_iterations

        executor = PolicyExecutor()

        # Always return tool calls so it loops until policy stops it
        tc = _make_tool_call("looping_tool")
        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]

        loop, _ = _make_loop(max_iterations=100, executor=executor)
        loop.provider.chat = AsyncMock(return_value=tool_resp)
        loop.tools.execute = AsyncMock(return_value="ok")

        final, tools_used, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        # Should have stopped at 3 iterations, not 100
        assert len(tools_used) == 3
        assert executor.call_count > 0

    async def test_executor_on_limit_reached_custom_message(self) -> None:
        """When the executor has on_limit_reached(), it provides the limit message."""
        from exoclaw.executor import DirectExecutor

        class PolicyExecutor(DirectExecutor):
            async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
                return iteration < 1

            async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
                return f"Custom limit hit at {iteration} iterations, used: {', '.join(tools_used)}"

        executor = PolicyExecutor()

        tc = _make_tool_call("my_tool")
        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]

        loop, _ = _make_loop(max_iterations=100, executor=executor)
        loop.provider.chat = AsyncMock(return_value=tool_resp)
        loop.tools.execute = AsyncMock(return_value="ok")

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert "Custom limit hit" in final
        assert "my_tool" in final

    async def test_default_behavior_without_executor_methods(self) -> None:
        """Without should_continue/on_limit_reached, falls back to max_iterations."""
        loop, _ = _make_loop(max_iterations=2)

        tc = _make_tool_call("dummy")
        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]

        loop.provider.chat = AsyncMock(return_value=tool_resp)
        loop.tools.execute = AsyncMock(return_value="ok")

        final, tools_used, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert len(tools_used) == 2
        assert "maximum" in final.lower()

    async def test_should_continue_receives_tool_names(self) -> None:
        """should_continue receives accumulated tool names for pattern detection."""
        from exoclaw.executor import DirectExecutor

        captured_tools: list[list[str]] = []

        class SpyExecutor(DirectExecutor):
            async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
                captured_tools.append(list(tools_used))
                return iteration < 3

        executor = SpyExecutor()

        tc1 = _make_tool_call("search", call_id="tc1")
        tc2 = _make_tool_call("read_file", call_id="tc2")
        tool_resp1 = _make_response(has_tool_calls=True)
        tool_resp1.tool_calls = [tc1]
        tool_resp2 = _make_response(has_tool_calls=True)
        tool_resp2.tool_calls = [tc2]
        final_resp = _make_response(content="done")

        loop, _ = _make_loop(executor=executor)
        loop.provider.chat = AsyncMock(side_effect=[tool_resp1, tool_resp2, final_resp])
        loop.tools.execute = AsyncMock(return_value="ok")

        await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        # First check: no tools yet; second: ["search"]; third: ["search", "read_file"]
        assert captured_tools[0] == []
        assert captured_tools[1] == ["search"]
        assert captured_tools[2] == ["search", "read_file"]
