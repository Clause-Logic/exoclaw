"""Channel protocol — the only thing ChannelManager depends on."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from exoclaw.bus.events import OutboundMessage

if TYPE_CHECKING:
    from exoclaw.bus.protocol import Bus


@runtime_checkable
class Channel(Protocol):
    """
    Protocol for chat channel implementations.

    Any object with these three methods and a name attribute satisfies
    this protocol — no inheritance from BaseChannel required.
    """

    name: str

    async def start(self, bus: "Bus") -> None:
        """Connect to the platform and begin receiving messages."""
        ...

    async def stop(self) -> None:
        """Disconnect and release resources."""
        ...

    async def send(self, msg: OutboundMessage) -> None:
        """Deliver an outbound message to the platform."""
        ...
