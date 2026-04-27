"""MessageBus round-trip tests on MicroPython.

Pure-Python (no pytest) — runs under ``tests/_micropython_runner/run.py``.
Mirrors the bus behaviour exoclaw expects: a channel ``publish_inbound``
delivers to the agent's ``consume_inbound``, and the same shape works
for the outbound direction.

These tests prove ``_compat.make_async_queue`` (which on MicroPython is
the ``_AsyncQueue`` shim — uasyncio doesn't ship ``Queue``) is a
drop-in for ``asyncio.Queue`` from MessageBus's perspective.
"""

import asyncio

from exoclaw.bus.events import InboundMessage, OutboundMessage
from exoclaw.bus.queue import MessageBus


def _run(coro):
    """Helper: ``asyncio.run`` is on MP but we want one entry point
    so the per-test boilerplate stays small."""
    asyncio.run(coro)


def test_inbound_round_trip():
    bus = MessageBus()

    async def _go():
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="hello")
        await bus.publish_inbound(msg)
        got = await bus.consume_inbound()
        assert got is msg, "inbound round-trip lost identity"
        assert got.session_key == "telegram:c1"

    _run(_go())


def test_outbound_round_trip():
    bus = MessageBus()

    async def _go():
        out = OutboundMessage(channel="discord", chat_id="c2", content="reply")
        await bus.publish_outbound(out)
        got = await bus.consume_outbound()
        assert got is out

    _run(_go())


def test_inbound_hook_diverts_publish():
    """When ``set_inbound_hook`` is installed, ``publish_inbound``
    forwards to the hook instead of putting on the queue. Same
    durable-executor wiring path as on CPython."""
    bus = MessageBus()
    seen = []

    async def _hook(msg):
        seen.append(msg)

    bus.set_inbound_hook(_hook)

    async def _go():
        msg = InboundMessage(channel="x", sender_id="u", chat_id="c", content="m")
        await bus.publish_inbound(msg)
        # Queue should be untouched.
        assert bus.inbound.empty(), "hook should bypass the queue"
        assert seen == [msg]

    _run(_go())


def test_inbound_hook_clear_restores_queue_path():
    """Passing ``None`` to ``set_inbound_hook`` restores the
    queue-based path. Round-trip works again."""
    bus = MessageBus()

    async def _hook(msg):
        raise AssertionError("hook should not fire after clear")

    bus.set_inbound_hook(_hook)
    bus.set_inbound_hook(None)

    async def _go():
        msg = InboundMessage(channel="x", sender_id="u", chat_id="c", content="m")
        await bus.publish_inbound(msg)
        got = await bus.consume_inbound()
        assert got is msg

    _run(_go())
