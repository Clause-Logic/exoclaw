"""Tests for the Executor protocol and DirectExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.executor import DirectExecutor, Executor


class TestDirectExecutorChat:
    async def test_delegates_to_provider(self) -> None:
        executor = DirectExecutor()
        provider = MagicMock()
        expected = MagicMock()
        provider.chat = AsyncMock(return_value=expected)

        result = await executor.chat(
            provider,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            reasoning_effort=None,
        )

        assert result is expected
        provider.chat.assert_awaited_once_with(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            reasoning_effort=None,
        )


class TestDirectExecutorExecuteTool:
    async def test_delegates_to_registry(self) -> None:
        executor = DirectExecutor()
        registry = MagicMock()
        registry.execute = AsyncMock(return_value="tool result")
        ctx = ToolContext(session_key="s:1", channel="cli", chat_id="1")

        result = await executor.execute_tool(registry, "my_tool", {"x": 1}, ctx)

        assert result == "tool result"
        registry.execute.assert_awaited_once_with("my_tool", {"x": 1}, ctx)

    async def test_none_ctx(self) -> None:
        executor = DirectExecutor()
        registry = MagicMock()
        registry.execute = AsyncMock(return_value="ok")

        result = await executor.execute_tool(registry, "t", {}, None)

        assert result == "ok"
        registry.execute.assert_awaited_once_with("t", {}, None)


class TestDirectExecutorBuildPrompt:
    async def test_delegates_to_conversation(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        expected = [{"role": "system", "content": "hi"}]
        conversation.build_prompt = AsyncMock(return_value=expected)

        result = await executor.build_prompt(
            conversation,
            "session:1",
            "hello",
            channel="cli",
            chat_id="1",
        )

        assert result is expected
        conversation.build_prompt.assert_awaited_once_with(
            "session:1",
            "hello",
            channel="cli",
            chat_id="1",
            media=None,
            plugin_context=None,
        )


class TestDirectExecutorRecord:
    async def test_delegates_to_conversation(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        conversation.record = AsyncMock()
        msgs = [{"role": "assistant", "content": "done"}]

        await executor.record(conversation, "session:1", msgs)

        conversation.record.assert_awaited_once_with("session:1", msgs)


class TestDirectExecutorClear:
    async def test_delegates_to_conversation(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        conversation.clear = AsyncMock(return_value=True)

        result = await executor.clear(conversation, "session:1")

        assert result is True
        conversation.clear.assert_awaited_once_with("session:1")


class TestDirectExecutorRunHook:
    async def test_calls_hook_function(self) -> None:
        executor = DirectExecutor()
        hook = AsyncMock(return_value="extra context")

        result = await executor.run_hook(hook, "arg1", "arg2", kw="val")

        assert result == "extra context"
        hook.assert_awaited_once_with("arg1", "arg2", kw="val")

    async def test_returns_none_from_hook(self) -> None:
        executor = DirectExecutor()
        hook = AsyncMock(return_value=None)

        result = await executor.run_hook(hook)

        assert result is None


class TestExecutorProtocol:
    def test_direct_executor_satisfies_protocol(self) -> None:
        assert isinstance(DirectExecutor(), Executor)


class TestCustomExecutorInLoop:
    """Verify the loop routes I/O through a custom executor."""

    async def test_chat_routed_through_executor(self) -> None:
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test"
        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        # Custom executor that records calls
        executor = MagicMock(spec=DirectExecutor)
        response = MagicMock()
        response.has_tool_calls = False
        response.content = "hello"
        response.finish_reason = "stop"
        response.reasoning_content = None
        response.thinking_blocks = None
        executor.chat = AsyncMock(return_value=response)
        executor.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        executor.record = AsyncMock()
        executor.run_turn = AsyncMock(return_value=None)

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            executor=executor,
        )

        from exoclaw.bus.events import InboundMessage

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        result = await loop._process_message(msg)

        assert result is not None
        assert result.content == "hello"
        # Verify executor.chat was called, not provider.chat directly
        executor.chat.assert_awaited_once()
        provider.chat.assert_not_called()

    async def test_clear_routed_through_executor(self) -> None:
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.bus.events import InboundMessage
        from exoclaw.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test"
        conversation = MagicMock()

        executor = MagicMock(spec=DirectExecutor)
        executor.clear = AsyncMock(return_value=True)

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            executor=executor,
        )

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="/new")
        result = await loop._process_message(msg)

        assert "New session" in result.content
        executor.clear.assert_awaited_once()
        # conversation.clear should NOT be called directly
        conversation.clear.assert_not_called()

    async def test_execute_tool_routed_through_executor(self) -> None:
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.bus.events import InboundMessage
        from exoclaw.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test"
        conversation = MagicMock()

        # Build an executor that returns tool calls on first chat, then final on second
        executor = MagicMock(spec=DirectExecutor)

        tc = MagicMock()
        tc.id = "call1"
        tc.name = "my_tool"
        tc.arguments = {"q": "test"}

        tool_resp = MagicMock()
        tool_resp.has_tool_calls = True
        tool_resp.content = None
        tool_resp.tool_calls = [tc]
        tool_resp.reasoning_content = None
        tool_resp.thinking_blocks = None

        final_resp = MagicMock()
        final_resp.has_tool_calls = False
        final_resp.content = "done"
        final_resp.finish_reason = "stop"
        final_resp.reasoning_content = None
        final_resp.thinking_blocks = None

        executor.chat = AsyncMock(side_effect=[tool_resp, final_resp])
        executor.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        executor.record = AsyncMock()
        executor.execute_tool = AsyncMock(return_value="tool output")
        executor.run_turn = AsyncMock(return_value=None)

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            executor=executor,
        )
        # Register a dummy tool so get_definitions returns something
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "test"
        tool.parameters = {}
        tool.sent_in_turn = False
        loop.tools.register(tool)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="do it")
        result = await loop._process_message(msg)

        assert result.content == "done"
        executor.execute_tool.assert_awaited_once()
