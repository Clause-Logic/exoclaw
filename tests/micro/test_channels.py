"""ChannelManager coordination on MicroPython.

The manager wires a set of ``Channel``-protocol objects to the bus:
``start_all`` brings them up + spawns the outbound-dispatch task,
``stop_all`` cancels the dispatch task and tears down each channel,
``_dispatch_outbound`` routes each ``OutboundMessage`` to its named
channel.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

import asyncio

from exoclaw.bus.events import OutboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.channels.manager import ChannelManager


class _RecordingChannel:
    """Channel-protocol stub: records start / stop / send calls.

    ``start`` accepts a ``Bus`` and records nothing on it (real
    channels would subscribe to inbound deliveries from the platform
    and publish them to the bus); for tests we just record the
    lifecycle hits."""

    def __init__(self, name):
        self.name = name
        self.started = False
        self.stopped = False
        self.sent = []

    async def start(self, bus):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def send(self, msg):
        self.sent.append(msg)


def test_register_after_construction():
    """``register`` adds a channel post-init — same path the dynamic
    plugin loader uses."""
    bus = MessageBus()
    mgr = ChannelManager([], bus)
    assert mgr.channels == {}
    ch = _RecordingChannel("late")
    mgr.register(ch)
    assert mgr.get_channel("late") is ch


def test_start_all_with_no_channels_warns():
    """Empty channel list short-circuits to a ``no_channels`` log
    line — covers the early-return branch."""
    bus = MessageBus()
    mgr = ChannelManager([], bus)

    async def _go():
        await mgr.start_all()
        # No dispatch task spawned when empty.
        assert mgr._dispatch_task is None

    asyncio.run(_go())


def test_start_all_starts_each_channel():
    """Every registered channel's ``start`` gets called with the bus.
    The dispatch task is spawned to drain outbound."""
    bus = MessageBus()
    a = _RecordingChannel("a")
    b = _RecordingChannel("b")
    mgr = ChannelManager([a, b], bus)

    async def _go():
        await mgr.start_all()
        assert a.started is True
        assert b.started is True
        assert mgr._dispatch_task is not None
        # Cancel the dispatch task so the test loop exits.
        await mgr.stop_all()

    asyncio.run(_go())


def test_stop_all_stops_each_channel():
    """``stop_all`` calls ``stop`` on every channel + cancels the
    dispatch task."""
    bus = MessageBus()
    a = _RecordingChannel("a")
    mgr = ChannelManager([a], bus)

    async def _go():
        await mgr.start_all()
        await mgr.stop_all()
        assert a.stopped is True
        # Dispatch task was cancelled — accessing its result raises
        # CancelledError, but we don't await it again here.

    asyncio.run(_go())


def test_dispatch_routes_outbound_to_named_channel():
    """An outbound message published on the bus reaches the channel
    whose ``name`` matches the message's ``channel`` field."""
    bus = MessageBus()
    a = _RecordingChannel("telegram")
    b = _RecordingChannel("discord")
    mgr = ChannelManager([a, b], bus)

    async def _go():
        await mgr.start_all()
        # Publish an outbound for telegram. The dispatcher should
        # route it to the telegram channel only.
        out = OutboundMessage(channel="telegram", chat_id="c1", content="hi")
        await bus.publish_outbound(out)
        # Yield long enough for the dispatch task to drain the queue
        # (consume_outbound has a 1s timeout — sleep a beat is enough).
        await asyncio.sleep(0.05)
        assert a.sent == [out], "expected message routed to telegram"
        assert b.sent == [], "discord should not have received the message"
        await mgr.stop_all()

    asyncio.run(_go())


def test_dispatch_unknown_channel_logged():
    """Outbound for a channel that isn't registered logs ``unknown_channel``
    and doesn't raise."""
    bus = MessageBus()
    a = _RecordingChannel("known")
    mgr = ChannelManager([a], bus)

    async def _go():
        await mgr.start_all()
        out = OutboundMessage(channel="ghost", chat_id="c", content="hi")
        await bus.publish_outbound(out)
        await asyncio.sleep(0.05)
        # No channel got it; no exception either.
        assert a.sent == []
        await mgr.stop_all()

    asyncio.run(_go())


def test_channel_start_exception_logged_not_raised():
    """A channel whose ``start`` raises shouldn't crash the
    manager — the exception is logged (``channel_start_error``)
    and the manager continues with whatever channels did start."""

    class _BrokenStart:
        name = "broken"
        started = False

        async def start(self, bus):
            raise RuntimeError("start failed")

        async def stop(self):
            pass

        async def send(self, msg):
            pass

    bus = MessageBus()
    a = _RecordingChannel("good")
    b = _BrokenStart()
    mgr = ChannelManager([a, b], bus)

    async def _go():
        await mgr.start_all()
        # The good channel started; the broken one didn't crash the manager.
        assert a.started is True
        await mgr.stop_all()

    asyncio.run(_go())


def test_channel_stop_exception_logged_not_raised():
    """``stop_all`` ignores exceptions from individual channel
    ``stop`` calls — same defensive posture as ``start_all``."""

    class _BrokenStop:
        name = "broken"

        async def start(self, bus):
            pass

        async def stop(self):
            raise RuntimeError("stop failed")

        async def send(self, msg):
            pass

    bus = MessageBus()
    a = _RecordingChannel("good")
    b = _BrokenStop()
    mgr = ChannelManager([a, b], bus)

    async def _go():
        await mgr.start_all()
        # Should not raise.
        await mgr.stop_all()
        assert a.stopped is True

    asyncio.run(_go())


def test_channel_send_exception_logged_not_raised():
    """A channel whose ``send`` raises during outbound dispatch
    doesn't crash the manager — logged as ``outbound_send_error``
    and the dispatch loop continues."""

    class _BrokenSend:
        name = "bad"

        async def start(self, bus):
            pass

        async def stop(self):
            pass

        async def send(self, msg):
            raise RuntimeError("send failed")

    bus = MessageBus()
    mgr = ChannelManager([_BrokenSend()], bus)

    async def _go():
        await mgr.start_all()
        out = OutboundMessage(channel="bad", chat_id="x", content="hi")
        await bus.publish_outbound(out)
        # Yield to let dispatch try to send. No exception should propagate.
        await asyncio.sleep(0.05)
        # Subsequent publishes still work — the loop didn't die.
        await bus.publish_outbound(OutboundMessage(channel="bad", chat_id="x", content="hi2"))
        await asyncio.sleep(0.05)
        await mgr.stop_all()

    asyncio.run(_go())


def test_filter_tool_hints():
    """When ``filter_tool_hints=True`` the manager drops outbound
    messages tagged with ``_tool_hint`` in metadata. Used by some
    plugins to prevent echoing the LLM's tool intent back as user
    chat noise."""
    bus = MessageBus()
    a = _RecordingChannel("c")
    mgr = ChannelManager([a], bus, filter_tool_hints=True)

    async def _go():
        await mgr.start_all()
        # Hint message — should be filtered.
        hint = OutboundMessage(
            channel="c", chat_id="x", content="...", metadata={"_tool_hint": True}
        )
        await bus.publish_outbound(hint)
        # Plain message — should pass.
        plain = OutboundMessage(channel="c", chat_id="x", content="real reply")
        await bus.publish_outbound(plain)
        await asyncio.sleep(0.05)
        assert a.sent == [plain]
        await mgr.stop_all()

    asyncio.run(_go())
