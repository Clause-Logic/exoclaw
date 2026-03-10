"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from exoclaw.agent.loop import AgentLoop
    from exoclaw.bus.queue import MessageBus


def _make_loop() -> tuple["AgentLoop", "MessageBus"]:
    """Create a minimal AgentLoop with mocked dependencies."""
    from exoclaw.agent.loop import AgentLoop
    from exoclaw.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(bus=bus, provider=provider, conversation=MagicMock())
    return loop, bus


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self) -> None:
        from exoclaw.bus.events import InboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self) -> None:
        from exoclaw.bus.events import InboundMessage

        loop, bus = _make_loop()
        cancelled = asyncio.Event()

        async def slow_task() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = [task]

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)

        assert cancelled.is_set()
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self) -> None:
        from exoclaw.bus.events import InboundMessage

        loop, bus = _make_loop()
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx: int) -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = tasks

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)

        assert all(e.is_set() for e in events)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "2 task" in out.content


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self) -> None:
        from exoclaw.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_processing_lock_serializes(self) -> None:
        from exoclaw.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []

        async def mock_process(m: object, **kwargs: object) -> OutboundMessage:
            order.append(f"start-{m.content}")  # type: ignore[union-attr]
            await asyncio.sleep(0.05)
            order.append(f"end-{m.content}")  # type: ignore[union-attr]
            return OutboundMessage(channel="test", chat_id="c1", content=m.content)  # type: ignore[union-attr]

        loop._process_message = mock_process
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]


class TestRunCancellation:
    @pytest.mark.asyncio
    async def test_run_cancels_cleanly_with_active_dispatch(self) -> None:
        """When run() is cancelled, it must also cancel any in-flight _dispatch() child tasks."""
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.bus.events import InboundMessage
        from exoclaw.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        loop = AgentLoop(bus=bus, provider=provider, conversation=MagicMock())

        dispatch_started = asyncio.Event()
        dispatch_cancelled = asyncio.Event()

        async def slow_dispatch(msg: object) -> None:
            dispatch_started.set()
            try:
                await asyncio.sleep(60)  # simulates a hung LLM call
            except asyncio.CancelledError:
                dispatch_cancelled.set()
                raise

        loop._dispatch = slow_dispatch  # type: ignore[method-assign]

        run_task = asyncio.create_task(loop.run())

        # Publish a message so _dispatch starts
        await bus.publish_inbound(
            InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        )
        await asyncio.wait_for(dispatch_started.wait(), timeout=2.0)

        # Cancel run() — the child dispatch task must also be cancelled
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)

        # Give the event loop one tick to propagate cancellation to child tasks
        await asyncio.sleep(0)

        assert dispatch_cancelled.is_set(), (
            "run() was cancelled but the in-flight _dispatch() task was not cancelled — "
            "child tasks are leaked and will hang the process"
        )
