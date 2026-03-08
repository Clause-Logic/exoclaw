"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
from loguru import logger

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

    def __init__(self, channels: list[Channel], bus: Bus):
        self.bus = bus
        self.channels: dict[str, Channel] = {ch.name: ch for ch in channels}
        self._dispatch_task: asyncio.Task | None = None

    def register(self, channel: Channel) -> None:
        """Register a channel after construction."""
        self.channels[channel.name] = channel

    async def _start_channel(self, name: str, channel: Channel) -> None:
        try:
            await channel.start(self.bus)
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)

    async def start_all(self) -> None:
        if not self.channels:
            logger.warning("No channels enabled")
            return

        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        tasks = [
            asyncio.create_task(self._start_channel(name, ch))
            for name, ch in self.channels.items()
        ]
        logger.info("Starting channels: {}", list(self.channels))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        logger.info("Stopping all channels...")

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except Exception as e:
                logger.error("Error stopping {}: {}", name, e)

    async def _dispatch_outbound(self) -> None:
        logger.info("Outbound dispatcher started")

        while True:
            try:
                msg: OutboundMessage = await asyncio.wait_for(
                    self.bus.consume_outbound(),
                    timeout=1.0,
                )

                if msg.metadata.get("_progress"):
                    # Progress filtering is a policy concern — handled by caller
                    pass

                channel = self.channels.get(msg.channel)
                if channel:
                    try:
                        await channel.send(msg)
                    except Exception as e:
                        logger.error("Error sending to {}: {}", msg.channel, e)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_channel(self, name: str) -> Channel | None:
        return self.channels.get(name)

