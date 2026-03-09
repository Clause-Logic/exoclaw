"""
Exoclaw — the composition root.

Usage:

    import asyncio
    from exoclaw import Exoclaw

    app = Exoclaw(
        provider=MyProvider(),
        conversation=MyConversation(),
        channels=[MyChannel(...)],
    )
    asyncio.run(app.run())
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import Tool
from exoclaw.bus.protocol import Bus
from exoclaw.channels.manager import ChannelManager
from exoclaw.channels.protocol import Channel

if TYPE_CHECKING:
    from exoclaw.providers.protocol import LLMProvider


class Exoclaw:
    """
    Wires together all exoclaw components and runs the event loop.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        conversation: Conversation,
        channels: list[Channel] | None = None,
        tools: list[Tool] | None = None,
        bus: Bus | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_iterations: int = 40,
        reasoning_effort: str | None = None,
    ):
        self.provider = provider
        self.conversation = conversation
        self.channels = channels or []
        self.tools = tools or []
        self.bus = bus
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.reasoning_effort = reasoning_effort

    def _build(self) -> tuple[Any, Any, Any]:
        """Instantiate all internal components. Called once at run time."""
        from exoclaw.agent.loop import AgentLoop

        if self.bus is not None:
            bus = self.bus
        else:
            from exoclaw.bus.queue import MessageBus
            bus = MessageBus()

        model = self.model or self.provider.get_default_model()

        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            conversation=self.conversation,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_iterations=self.max_iterations,
            reasoning_effort=self.reasoning_effort,
            tools=self.tools,
        )

        channel_manager = ChannelManager(self.channels, bus)

        return bus, agent, channel_manager

    async def run(self) -> None:
        """Start all components and run until interrupted."""
        bus, agent, channel_manager = self._build()

        logger.info("Exoclaw starting")

        try:
            await asyncio.gather(
                agent.run(),
                channel_manager.start_all(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutting down...")
        finally:
            agent.stop()
            await channel_manager.stop_all()
