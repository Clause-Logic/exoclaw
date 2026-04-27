"""Async message queue for decoupled channel-agent communication."""

from typing import Any, Awaitable, Callable

from exoclaw._compat import make_async_queue
from exoclaw.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.

    When a durable executor is in play, ``AgentLoop`` installs an
    inbound hook via ``set_inbound_hook``. After that, channels still
    call ``publish_inbound`` the same way, but the message is handed
    off to the executor's durable store instead of the in-memory queue
    — closing the window where a crash between "channel received" and
    "agent began processing" would lose the message.
    """

    def __init__(self) -> None:
        # ``make_async_queue`` returns ``asyncio.Queue`` on CPython,
        # the ``_AsyncQueue`` shim on MicroPython (uasyncio doesn't
        # ship ``Queue``). Same ``put`` / ``get`` surface either way.
        # ``Any`` type because the shape differs by runtime — exoclaw
        # treats both as opaque queue-likes.
        self.inbound: Any = make_async_queue()
        self.outbound: Any = make_async_queue()
        self._inbound_hook: Callable[[InboundMessage], Awaitable[None]] | None = None

    def set_inbound_hook(self, handler: Callable[[InboundMessage], Awaitable[None]] | None) -> None:
        """Install (or clear) a durable-inbound handler.

        When set, ``publish_inbound`` forwards to this handler instead
        of putting the message on the asyncio queue. Intended for
        durable executors: the handler starts a workflow so the message
        is persisted before ``publish_inbound`` returns.

        Passing ``None`` restores the in-memory queue path.
        """
        self._inbound_hook = handler

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        if self._inbound_hook is not None:
            await self._inbound_hook(msg)
            return
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()
