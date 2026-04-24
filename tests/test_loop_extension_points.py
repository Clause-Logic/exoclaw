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
# IterationPolicy — pluggable termination strategy
# ---------------------------------------------------------------------------


class TestIterationPolicy:
    async def test_policy_controls_loop(self) -> None:
        """When an IterationPolicy is provided, it replaces the max_iterations check."""

        class StopAt3:
            def __init__(self) -> None:
                self.call_count = 0

            async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
                self.call_count += 1
                return iteration < 3

            async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
                return "policy stopped"

        policy = StopAt3()

        tc = _make_tool_call("looping_tool")
        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]

        loop, _ = _make_loop(max_iterations=100, iteration_policy=policy)
        loop.provider.chat = AsyncMock(return_value=tool_resp)
        loop.tools.execute = AsyncMock(return_value="ok")

        final, tools_used, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        # Should have stopped at 3 iterations, not 100
        assert len(tools_used) == 3
        assert policy.call_count > 0

    async def test_policy_custom_limit_message(self) -> None:
        """on_limit_reached() provides the termination message."""

        class CustomMessage:
            async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
                return iteration < 1

            async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
                return f"Custom limit hit at {iteration} iterations, used: {', '.join(tools_used)}"

        tc = _make_tool_call("my_tool")
        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]

        loop, _ = _make_loop(max_iterations=100, iteration_policy=CustomMessage())
        loop.provider.chat = AsyncMock(return_value=tool_resp)
        loop.tools.execute = AsyncMock(return_value="ok")

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert "Custom limit hit" in final
        assert "my_tool" in final

    async def test_default_behavior_without_policy(self) -> None:
        """Without an IterationPolicy, falls back to max_iterations."""
        loop, _ = _make_loop(max_iterations=2)

        tc = _make_tool_call("dummy")
        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]

        loop.provider.chat = AsyncMock(return_value=tool_resp)
        loop.tools.execute = AsyncMock(return_value="ok")

        final, tools_used, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert len(tools_used) == 2
        assert "maximum" in final.lower()

    async def test_policy_receives_tool_names(self) -> None:
        """should_continue receives accumulated tool names for pattern detection."""
        captured_tools: list[list[str]] = []

        class SpyPolicy:
            async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
                captured_tools.append(list(tools_used))
                return iteration < 3

            async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
                return "stopped"

        tc1 = _make_tool_call("search", call_id="tc1")
        tc2 = _make_tool_call("read_file", call_id="tc2")
        tool_resp1 = _make_response(has_tool_calls=True)
        tool_resp1.tool_calls = [tc1]
        tool_resp2 = _make_response(has_tool_calls=True)
        tool_resp2.tool_calls = [tc2]
        final_resp = _make_response(content="done")

        loop, _ = _make_loop(iteration_policy=SpyPolicy())
        loop.provider.chat = AsyncMock(side_effect=[tool_resp1, tool_resp2, final_resp])
        loop.tools.execute = AsyncMock(return_value="ok")

        await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        # First check: no tools yet; second: ["search"]; third: ["search", "read_file"]
        assert captured_tools[0] == []
        assert captured_tools[1] == ["search"]
        assert captured_tools[2] == ["search", "read_file"]

    async def test_policy_composes_with_any_executor(self) -> None:
        """IterationPolicy works independently of the executor — they compose orthogonally."""
        from exoclaw.executor import DirectExecutor

        class CustomExecutor(DirectExecutor):
            """Simulates a Temporal/custom executor — no iteration methods."""

            pass

        class StopAt2:
            async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
                return iteration < 2

            async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
                return "policy stopped with custom executor"

        tc = _make_tool_call("tool_a")
        tool_resp = _make_response(has_tool_calls=True)
        tool_resp.tool_calls = [tc]

        loop, _ = _make_loop(executor=CustomExecutor(), iteration_policy=StopAt2())
        loop.provider.chat = AsyncMock(return_value=tool_resp)
        loop.tools.execute = AsyncMock(return_value="ok")

        final, tools_used, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert len(tools_used) == 2
        assert "policy stopped with custom executor" in final


# ---------------------------------------------------------------------------
# run_turn delegation
# ---------------------------------------------------------------------------


class TestRunTurnDelegation:
    async def test_process_turn_delegates_to_executor_run_turn(self) -> None:
        """When executor.run_turn() returns a result, process_turn uses it."""
        from exoclaw.executor import DirectExecutor

        class DurableExecutor(DirectExecutor):
            async def run_turn(
                self,
                loop: object,
                session_id: str,
                message: str,
                **kwargs: object,
            ) -> tuple[str | None, list[dict[str, object]]] | None:
                return ("durable result", [{"role": "assistant", "content": "durable result"}])

        loop, _ = _make_loop(executor=DurableExecutor())

        content, msgs = await loop.process_turn("sess1", "hello")

        assert content == "durable result"
        assert msgs == [{"role": "assistant", "content": "durable result"}]
        # build_prompt should NOT have been called — delegation skipped inline path
        loop.conversation.build_prompt.assert_not_called()
        loop.conversation.record.assert_not_called()

    async def test_process_turn_falls_back_when_run_turn_returns_none(self) -> None:
        """When executor.run_turn() returns None, process_turn uses the inline path."""
        loop, _ = _make_loop()  # DirectExecutor.run_turn returns None

        content, msgs = await loop.process_turn("sess1", "hello")

        # Inline path was used — build_prompt and record should be called
        loop.conversation.build_prompt.assert_called_once()
        loop.conversation.record.assert_called_once()


# ---------------------------------------------------------------------------
# Executor-owned response send — publish_response / handles_response_send
# ---------------------------------------------------------------------------


class TestExecutorOwnsResponseSend:
    async def test_process_message_returns_none_when_executor_owns_send(self) -> None:
        """When the executor advertises ``handles_response_send=True`` and the
        caller requests it via ``publish_response=True``, ``_process_message``
        returns ``None`` so the caller knows the reply has already been
        dispatched by the executor."""
        from exoclaw.executor import DirectExecutor

        class OwningExecutor(DirectExecutor):
            handles_response_send: bool = True

            async def run_turn(
                self,
                loop: object,
                session_id: str,
                message: str,
                **kwargs: object,
            ) -> tuple[str | None, list[dict[str, object]]] | None:
                return ("final", [{"role": "assistant", "content": "final"}])

        loop, _ = _make_loop(executor=OwningExecutor())
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hi")

        response = await loop._process_message(msg, publish_response=True)

        assert response is None

    async def test_process_message_returns_outbound_when_publish_response_false(self) -> None:
        """When ``publish_response=False`` the caller is asking for the reply
        content back (e.g. ``process_direct``). The executor's
        ``handles_response_send`` is not consulted and the ``OutboundMessage``
        is returned as normal."""
        from exoclaw.executor import DirectExecutor

        class OwningExecutor(DirectExecutor):
            handles_response_send: bool = True

            async def run_turn(
                self,
                loop: object,
                session_id: str,
                message: str,
                **kwargs: object,
            ) -> tuple[str | None, list[dict[str, object]]] | None:
                return ("final", [{"role": "assistant", "content": "final"}])

        loop, _ = _make_loop(executor=OwningExecutor())
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="main", content="hi")

        response = await loop._process_message(msg, publish_response=False)

        assert response is not None
        assert response.content == "final"

    async def test_publish_response_forwarded_to_executor_run_turn(self) -> None:
        """``process_turn`` forwards ``publish_response`` to ``executor.run_turn``
        so the executor knows whether the caller is asking it to own the send."""
        from exoclaw.executor import DirectExecutor

        captured: dict[str, object] = {}

        class SpyExecutor(DirectExecutor):
            async def run_turn(
                self,
                loop: object,
                session_id: str,
                message: str,
                *,
                publish_response: bool = False,
                **kwargs: object,
            ) -> tuple[str | None, list[dict[str, object]]] | None:
                captured["publish_response"] = publish_response
                return ("ok", [])

        loop, _ = _make_loop(executor=SpyExecutor())

        await loop.process_turn("sess1", "hi", publish_response=True)

        assert captured["publish_response"] is True

    async def test_direct_executor_handles_response_send_default_false(self) -> None:
        """``DirectExecutor`` leaves the send to the caller — the core's
        default behavior when no executor opts in."""
        from exoclaw.executor import DirectExecutor

        assert DirectExecutor.handles_response_send is False


# ---------------------------------------------------------------------------
# Executor-owned inbound enqueue — handles_inbound_enqueue / set_inbound_hook
# ---------------------------------------------------------------------------


class TestExecutorOwnsInboundEnqueue:
    async def test_direct_executor_handles_inbound_enqueue_default_false(self) -> None:
        """``DirectExecutor`` leaves inbound on the asyncio queue — the
        core's default behavior when no durable executor opts in."""
        from exoclaw.executor import DirectExecutor

        assert DirectExecutor.handles_inbound_enqueue is False

    async def test_publish_inbound_falls_back_to_queue_when_no_hook(self) -> None:
        """No hook installed → ``publish_inbound`` drops the message on
        the asyncio queue for ``AgentLoop.run`` to consume, preserving
        pre-existing behavior for pass-through executors."""
        bus = MessageBus()
        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")

        await bus.publish_inbound(msg)

        assert bus.inbound.qsize() == 1
        assert bus.inbound.get_nowait() is msg

    async def test_publish_inbound_routes_to_hook_when_installed(self) -> None:
        """With a hook installed, ``publish_inbound`` forwards to it and
        does NOT enqueue on the asyncio queue — durable executors
        persist the message before ``publish_inbound`` returns."""
        bus = MessageBus()
        received: list[InboundMessage] = []

        async def hook(m: InboundMessage) -> None:
            received.append(m)

        bus.set_inbound_hook(hook)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        await bus.publish_inbound(msg)

        assert received == [msg]
        assert bus.inbound.qsize() == 0

    async def test_set_inbound_hook_none_restores_queue_path(self) -> None:
        """Clearing the hook restores the asyncio-queue path — lets
        tests and dynamic reconfiguration back out of durable wiring."""
        bus = MessageBus()
        calls: list[InboundMessage] = []

        async def hook(m: InboundMessage) -> None:
            calls.append(m)

        bus.set_inbound_hook(hook)
        bus.set_inbound_hook(None)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        await bus.publish_inbound(msg)

        assert calls == []
        assert bus.inbound.qsize() == 1

    async def test_agent_loop_wires_hook_for_durable_executor(self) -> None:
        """When the executor advertises ``handles_inbound_enqueue=True``,
        ``AgentLoop.__init__`` installs ``executor.enqueue_inbound`` as
        the bus's inbound hook. After that, every
        ``bus.publish_inbound`` goes to the executor instead of the
        asyncio queue."""
        from exoclaw.executor import DirectExecutor

        captured: list[InboundMessage] = []

        class DurableExecutor(DirectExecutor):
            handles_inbound_enqueue: bool = True

            async def enqueue_inbound(self, msg: InboundMessage) -> None:
                captured.append(msg)

        loop, bus = _make_loop(executor=DurableExecutor())

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        await bus.publish_inbound(msg)

        assert captured == [msg]
        assert bus.inbound.qsize() == 0
        # loop is the object under test — silence the unused-variable lint
        assert loop is not None

    async def test_agent_loop_skips_wiring_for_pass_through_executor(self) -> None:
        """A ``DirectExecutor`` (the default) leaves ``set_inbound_hook``
        unset so the asyncio-queue path keeps working — pre-existing
        deployments that don't use a durable executor are unaffected."""
        loop, bus = _make_loop()  # default DirectExecutor

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        await bus.publish_inbound(msg)

        assert bus.inbound.qsize() == 1
        assert loop is not None


# ---------------------------------------------------------------------------
# on_context_overflow — ContextWindowExceededError recovery
# ---------------------------------------------------------------------------


class TestContextOverflow:
    async def test_overflow_recovered_with_handler(self) -> None:
        """When provider raises ContextWindowExceededError and handler compacts, loop retries."""
        from exoclaw.providers.types import ContextWindowExceededError

        call_count = 0

        async def mock_chat(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContextWindowExceededError("prompt too long")
            return _make_response(content="recovered")

        async def compactor(messages: list[dict[str, object]]) -> list[dict[str, object]]:
            return [m for m in messages if m.get("role") == "user"]

        loop, _ = _make_loop(on_context_overflow=compactor)
        loop.provider.chat = AsyncMock(side_effect=mock_chat)

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert final == "recovered"
        assert call_count == 2

    async def test_overflow_unrecoverable_without_handler(self) -> None:
        """Without on_context_overflow, ContextWindowExceededError stops the loop."""
        from exoclaw.providers.types import ContextWindowExceededError

        loop, _ = _make_loop()
        loop.provider.chat = AsyncMock(side_effect=ContextWindowExceededError("too big"))

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert "context window" in final.lower()

    async def test_overflow_handler_returns_none_gives_up(self) -> None:
        """When handler returns None, loop gives up."""
        from exoclaw.providers.types import ContextWindowExceededError

        async def no_help(messages: list[dict[str, object]]) -> None:
            return None

        loop, _ = _make_loop(on_context_overflow=no_help)
        loop.provider.chat = AsyncMock(side_effect=ContextWindowExceededError("too big"))

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert "context window" in final.lower()

    async def test_overflow_during_tool_loop(self) -> None:
        """Overflow mid-tool-loop compacts and continues."""
        from exoclaw.providers.types import ContextWindowExceededError

        call_count = 0

        async def mock_chat(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ContextWindowExceededError("context full")
            if call_count == 1:
                tc = _make_tool_call("search")
                resp = _make_response(has_tool_calls=True)
                resp.tool_calls = [tc]
                return resp
            return _make_response(content="done after compact")

        async def compactor(messages: list[dict[str, object]]) -> list[dict[str, object]]:
            return [messages[0]]

        loop, _ = _make_loop(on_context_overflow=compactor)
        loop.provider.chat = AsyncMock(side_effect=mock_chat)
        loop.tools.execute = AsyncMock(return_value="result")

        final, _, _ = await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert final == "done after compact"
        assert call_count == 3
