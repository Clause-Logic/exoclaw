"""Bus protocol — the only bus surface external code should depend on."""

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from exoclaw.bus.events import InboundMessage, OutboundMessage


@runtime_checkable
class Bus(Protocol):
    async def publish_inbound(self, msg: InboundMessage) -> None: ...
    async def consume_inbound(self) -> InboundMessage: ...
    async def publish_outbound(self, msg: OutboundMessage) -> None: ...
    async def consume_outbound(self) -> OutboundMessage: ...

    def set_inbound_hook(self, handler: Callable[[InboundMessage], Awaitable[None]] | None) -> None:
        """Install (or clear) a durable-inbound handler.

        Optional on bus implementations. When set, ``publish_inbound``
        must forward to the handler instead of the in-memory queue so
        durable executors can persist messages before the channel's
        publish call returns.
        """
        ...
