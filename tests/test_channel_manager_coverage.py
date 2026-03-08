"""Tests for exoclaw/channels/manager.py coverage."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from exoclaw.bus.events import OutboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.channels.manager import ChannelManager


def _make_channel(name="test"):
    ch = MagicMock()
    ch.name = name
    ch.start = AsyncMock()
    ch.stop = AsyncMock()
    ch.send = AsyncMock()
    return ch


def _make_manager(channels=None):
    bus = MessageBus()
    chs = channels or []
    manager = ChannelManager(chs, bus)
    return manager, bus


class TestRegisterAndGetChannel:
    def test_register_adds_channel(self):
        manager, _ = _make_manager()
        ch = _make_channel("slack")
        manager.register(ch)
        assert manager.get_channel("slack") is ch

    def test_get_channel_missing_returns_none(self):
        manager, _ = _make_manager()
        assert manager.get_channel("nonexistent") is None

    def test_channels_from_constructor(self):
        ch = _make_channel("tg")
        manager, _ = _make_manager([ch])
        assert manager.get_channel("tg") is ch


class TestStartAll:
    async def test_start_all_no_channels(self):
        manager, _ = _make_manager()
        # Should return early without creating dispatch task
        await manager.start_all()
        assert manager._dispatch_task is None

    async def test_start_all_with_channels(self):
        ch = _make_channel("slack")
        manager, _ = _make_manager([ch])

        # start_all creates the dispatch task and starts channels — cancel after
        task = asyncio.create_task(manager.start_all())
        await asyncio.sleep(0.05)
        await manager.stop_all()
        await asyncio.wait_for(task, timeout=1.0)

        ch.start.assert_called_once()

    async def test_start_all_creates_dispatch_task(self):
        ch = _make_channel("slack")
        manager, _ = _make_manager([ch])

        task = asyncio.create_task(manager.start_all())
        await asyncio.sleep(0.05)
        assert manager._dispatch_task is not None
        await manager.stop_all()
        await asyncio.wait_for(task, timeout=1.0)


class TestStopAll:
    async def test_stop_all_no_dispatch_task(self):
        ch = _make_channel("slack")
        manager, _ = _make_manager([ch])
        # stop_all without ever calling start_all — no dispatch task
        await manager.stop_all()
        ch.stop.assert_called_once()

    async def test_stop_all_cancels_dispatch_task(self):
        ch = _make_channel("slack")
        manager, _ = _make_manager([ch])

        start_task = asyncio.create_task(manager.start_all())
        await asyncio.sleep(0.05)
        assert manager._dispatch_task is not None

        await manager.stop_all()
        await asyncio.wait_for(start_task, timeout=1.0)

        assert manager._dispatch_task.cancelled() or manager._dispatch_task.done()

    async def test_stop_all_channel_stop_raises(self):
        ch = _make_channel("slack")
        ch.stop = AsyncMock(side_effect=RuntimeError("stop error"))
        manager, _ = _make_manager([ch])
        # Should not raise even if channel.stop() fails
        await manager.stop_all()


class TestDispatchOutbound:
    async def test_message_routed_to_known_channel(self):
        ch = _make_channel("slack")
        manager, bus = _make_manager([ch])

        msg = OutboundMessage(channel="slack", chat_id="c1", content="hello")
        await bus.publish_outbound(msg)

        # Run dispatch loop briefly
        task = asyncio.create_task(manager._dispatch_outbound())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        ch.send.assert_called_once_with(msg)

    async def test_message_to_unknown_channel(self):
        ch = _make_channel("slack")
        manager, bus = _make_manager([ch])

        msg = OutboundMessage(channel="discord", chat_id="c1", content="hello")
        await bus.publish_outbound(msg)

        task = asyncio.create_task(manager._dispatch_outbound())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        ch.send.assert_not_called()

    async def test_progress_message_still_routed(self):
        ch = _make_channel("slack")
        manager, bus = _make_manager([ch])

        msg = OutboundMessage(channel="slack", chat_id="c1", content="...", metadata={"_progress": True})
        await bus.publish_outbound(msg)

        task = asyncio.create_task(manager._dispatch_outbound())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        ch.send.assert_called_once_with(msg)

    async def test_channel_send_raises_does_not_crash_loop(self):
        ch = _make_channel("slack")
        ch.send = AsyncMock(side_effect=RuntimeError("network error"))
        manager, bus = _make_manager([ch])

        msg = OutboundMessage(channel="slack", chat_id="c1", content="hello")
        await bus.publish_outbound(msg)

        task = asyncio.create_task(manager._dispatch_outbound())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The loop should have survived the exception
        ch.send.assert_called_once()

    async def test_dispatch_stops_on_cancelled_error(self):
        manager, _ = _make_manager()
        task = asyncio.create_task(manager._dispatch_outbound())
        await asyncio.sleep(0.05)
        task.cancel()
        # Should exit cleanly
        await asyncio.wait_for(task, timeout=1.0)


class TestStartChannel:
    async def test_start_channel_success(self):
        ch = _make_channel("slack")
        manager, bus = _make_manager()
        await manager._start_channel("slack", ch)
        ch.start.assert_called_once_with(bus)

    async def test_start_channel_exception_does_not_raise(self):
        ch = _make_channel("slack")
        ch.start = AsyncMock(side_effect=RuntimeError("connect failed"))
        manager, _ = _make_manager()
        # Should not raise
        await manager._start_channel("slack", ch)
