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


class TestDirectExecutorMessageManagement:
    def test_set_and_load(self) -> None:
        executor = DirectExecutor()
        msgs = [{"role": "user", "content": "hi"}]
        executor.set_messages(msgs)
        assert executor.load_messages() == msgs

    def test_append_and_load(self) -> None:
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "hi"}])
        executor.append_messages([{"role": "assistant", "content": "hello"}])
        loaded = executor.load_messages()
        assert len(loaded) == 2
        assert loaded[0]["role"] == "user"
        assert loaded[1]["role"] == "assistant"

    def test_load_returns_copy(self) -> None:
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "hi"}])
        loaded = executor.load_messages()
        loaded.append({"role": "assistant", "content": "extra"})
        assert len(executor.load_messages()) == 1

    def test_set_replaces(self) -> None:
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "first"}])
        executor.set_messages([{"role": "user", "content": "second"}])
        loaded = executor.load_messages()
        assert len(loaded) == 1
        assert loaded[0]["content"] == "second"

    def test_starts_empty(self) -> None:
        executor = DirectExecutor()
        assert executor.load_messages() == []

    def test_two_instances_isolate_messages(self) -> None:
        """Two executors in the same task must not share a buffer.

        The ContextVar is per-instance for exactly this reason — a
        module-level var would make a second executor reset the first's
        state and vice versa.
        """
        a = DirectExecutor()
        b = DirectExecutor()
        a.set_messages([{"role": "user", "content": "a"}])
        b.set_messages([{"role": "user", "content": "b"}])
        assert [m["content"] for m in a.load_messages()] == ["a"]
        assert [m["content"] for m in b.load_messages()] == ["b"]
        a.append_messages([{"role": "assistant", "content": "a2"}])
        assert [m["content"] for m in a.load_messages()] == ["a", "a2"]
        assert [m["content"] for m in b.load_messages()] == ["b"]

    async def test_concurrent_turns_isolate_messages(self) -> None:
        """The shared executor singleton must not leak messages across
        concurrent turns.

        Regression for the incident where a cron's ``Learn about
        Stephen`` turn fired mid-flight and its messages ended up
        pre-filling the context of a peer ``/agent/call`` turn, leading
        the peer's LLM to hallucinate it was running the cron's skill
        and cross-contaminating both session JSONLs.

        Each asyncio.Task gets its own ContextVar binding, so both
        turns must observe exactly the messages they themselves wrote.
        """
        import asyncio

        executor = DirectExecutor()
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def turn(label: str, out: dict[str, list[dict[str, object]]]) -> None:
            executor.set_messages([{"role": "user", "content": f"{label}:user"}])
            entered.set()
            await proceed.wait()
            executor.append_messages([{"role": "assistant", "content": f"{label}:asst"}])
            out[label] = executor.load_messages()

        results: dict[str, list[dict[str, object]]] = {}
        t1 = asyncio.create_task(turn("a", results))
        # Let t1 set its messages and suspend.
        await entered.wait()
        entered.clear()
        t2 = asyncio.create_task(turn("b", results))
        await entered.wait()
        # Both tasks have now set distinct message lists; release them.
        proceed.set()
        await asyncio.gather(t1, t2)

        assert [m["content"] for m in results["a"]] == ["a:user", "a:asst"]
        assert [m["content"] for m in results["b"]] == ["b:user", "b:asst"]


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

    async def test_build_prompt_seeds_messages(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        conversation.build_prompt = AsyncMock(return_value=msgs)

        await executor.build_prompt(conversation, "s:1", "hi", channel="cli", chat_id="1")

        assert executor.load_messages() == msgs


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
        _messages: list[dict[str, object]] = []
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
        executor.set_messages = MagicMock(
            side_effect=lambda m: _messages.clear() or _messages.extend(m)
        )
        executor.load_messages = MagicMock(side_effect=lambda: list(_messages))
        executor.append_messages = MagicMock(side_effect=lambda m: _messages.extend(m))

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
        _messages: list[dict[str, object]] = []
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
        executor.set_messages = MagicMock(
            side_effect=lambda m: _messages.clear() or _messages.extend(m)
        )
        executor.load_messages = MagicMock(side_effect=lambda: list(_messages))
        executor.append_messages = MagicMock(side_effect=lambda m: _messages.extend(m))

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
