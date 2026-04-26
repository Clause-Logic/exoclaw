"""Bus protocol — the only bus surface external code should depend on."""

from typing import Awaitable, Callable, Protocol, runtime_checkable

from exoclaw.bus.events import InboundMessage, OutboundMessage


@runtime_checkable
class Bus(Protocol):
    async def publish_inbound(self, msg: InboundMessage) -> None: ...
    async def consume_inbound(self) -> InboundMessage: ...
    async def publish_outbound(self, msg: OutboundMessage) -> None: ...
    async def consume_outbound(self) -> OutboundMessage: ...


@runtime_checkable
class InboundHookBus(Bus, Protocol):
    """Optional capability layered on top of ``Bus``: lets a durable
    executor install a handler that runs synchronously on every
    ``publish_inbound``.

    Kept separate from ``Bus`` so third-party bus implementations that
    don't need the capability aren't forced to stub a method just to
    satisfy ``@runtime_checkable`` / structural typing. Callers (e.g.
    ``AgentLoop``) use ``getattr(bus, "set_inbound_hook", None)`` or
    ``isinstance(bus, InboundHookBus)`` to check for support before
    calling it.
    """

    def set_inbound_hook(self, handler: Callable[[InboundMessage], Awaitable[None]] | None) -> None:
        """Install (or clear) a durable-inbound handler.

        When set, ``publish_inbound`` forwards to the handler instead
        of the in-memory queue so durable executors can persist
        messages before the channel's publish call returns.

        Passing ``None`` restores the default (in-memory) path.
        """
        ...
