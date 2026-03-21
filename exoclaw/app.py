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
from typing import TYPE_CHECKING

import structlog
from structlog.typing import FilteringBoundLogger

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import Tool
from exoclaw.bus.protocol import Bus
from exoclaw.channels.manager import ChannelManager
from exoclaw.channels.protocol import Channel
from exoclaw.executor import Executor
from exoclaw.iteration_policy import IterationPolicy

if TYPE_CHECKING:
    from exoclaw.agent.loop import AgentLoop
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
        iteration_policy: IterationPolicy | None = None,
        executor: Executor | None = None,
        logger: FilteringBoundLogger | None = None,
    ) -> None:
        self.provider = provider
        self.iteration_policy = iteration_policy
        self.executor = executor
        self.conversation = conversation
        self.channels = channels or []
        self.tools = tools or []
        self.bus = bus
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.reasoning_effort = reasoning_effort
        self._log: FilteringBoundLogger = logger or structlog.get_logger()

    def _build(self) -> tuple[Bus, AgentLoop, ChannelManager]:
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
            iteration_policy=self.iteration_policy,
            executor=self.executor,
            logger=self._log,
        )

        channel_manager = ChannelManager(self.channels, bus, logger=self._log)

        return bus, agent, channel_manager

    async def run(self) -> None:
        """Start all components and run until interrupted."""
        bus, agent, channel_manager = self._build()

        self._log.info("exoclaw_starting")

        try:
            await asyncio.gather(
                agent.run(),
                channel_manager.start_all(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._log.info("exoclaw_stopping")
        finally:
            agent.stop()
            await channel_manager.stop_all()
