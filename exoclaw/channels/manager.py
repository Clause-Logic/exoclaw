"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio

import structlog
from structlog.typing import FilteringBoundLogger

from exoclaw.bus.events import OutboundMessage
from exoclaw.bus.protocol import Bus
from exoclaw.channels.protocol import Channel


class ChannelManager:
    """
    Coordinates a set of Channel instances.

    Accepts any list of Channel-protocol objects — has no knowledge of
    specific platforms or their configuration. Platform wiring lives in
    ChannelFactory.
    """

    def __init__(
        self,
        channels: list[Channel],
        bus: Bus,
        filter_tool_hints: bool = False,
        logger: FilteringBoundLogger | None = None,
    ) -> None:
        self.bus = bus
        self.channels: dict[str, Channel] = {ch.name: ch for ch in channels}
        self._dispatch_task: asyncio.Task[None] | None = None
        self._filter_tool_hints = filter_tool_hints
        self._log: FilteringBoundLogger = logger or structlog.get_logger()

    def register(self, channel: Channel) -> None:
        """Register a channel after construction."""
        self.channels[channel.name] = channel

    async def _start_channel(self, name: str, channel: Channel) -> None:
        try:
            await channel.start(self.bus)
        except Exception:
            self._log.exception("channel_start_error", channel=name)

    async def start_all(self) -> None:
        if not self.channels:
            self._log.warning("no_channels")
            return

        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        tasks = [
            asyncio.create_task(self._start_channel(name, ch)) for name, ch in self.channels.items()
        ]
        self._log.info("channels_start", channels=list(self.channels))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        self._log.info("channels_stop")

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, channel in self.channels.items():
            try:
                await channel.stop()
            except Exception:
                self._log.exception("channel_stop_error", channel=name)

    async def _dispatch_outbound(self) -> None:

        while True:
            try:
                msg: OutboundMessage = await asyncio.wait_for(
                    self.bus.consume_outbound(),
                    timeout=1.0,
                )

                if self._filter_tool_hints and msg.metadata and msg.metadata.get("_tool_hint"):
                    continue

                channel = self.channels.get(msg.channel)
                if channel:
                    try:
                        await channel.send(msg)
                    except Exception:
                        self._log.exception("outbound_send_error", channel=msg.channel)
                else:
                    self._log.warning("unknown_channel", channel=msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_channel(self, name: str) -> Channel | None:
        return self.channels.get(name)
