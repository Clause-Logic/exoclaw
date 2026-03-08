"""
Nanobot — the composition root.

Usage (Python API):

    import asyncio
    from nanobot import Nanobot
    from nanobot_telegram import TelegramChannel

    app = Nanobot(
        provider=MyProvider(),
        channels=[TelegramChannel(...)],
        workspace="~/.nanobot/workspace",
    )
    asyncio.run(app.run())

Usage (CLI):

    nanobot run bot.py

where bot.py contains a module-level Nanobot instance named `app`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.tools.protocol import Tool
from nanobot.bus.protocol import Bus
from nanobot.channels.manager import ChannelManager
from nanobot.channels.protocol import Channel

if TYPE_CHECKING:
    from nanobot.providers.protocol import LLMProvider


class Nanobot:
    """
    Wires together all nanobot components and runs the event loop.

    The only required arguments are a provider and a list of channels.
    Everything else has a sensible default.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        channels: list[Channel] | None = None,
        tools: list[Tool] | None = None,
        bus: Bus | None = None,
        workspace: str | Path = "~/.nanobot/workspace",
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_iterations: int = 40,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        send_progress: bool = True,
        send_tool_hints: bool = False,
        enable_cron: bool = True,
        enable_heartbeat: bool = True,
        heartbeat_interval_s: int = 30 * 60,
    ):
        self.provider = provider
        self.channels = channels or []
        self.tools = tools or []
        self.bus = bus  # None = use default MessageBus
        self.workspace = Path(workspace).expanduser()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.send_progress = send_progress
        self.send_tool_hints = send_tool_hints
        self.enable_cron = enable_cron
        self.enable_heartbeat = enable_heartbeat
        self.heartbeat_interval_s = heartbeat_interval_s

    def _build(self):
        """Instantiate all internal components. Called once at run time."""
        from nanobot.agent.loop import AgentLoop
        from nanobot.config.schema import ChannelsConfig
        from nanobot.cron.service import CronService
        from nanobot.heartbeat.service import HeartbeatService
        from nanobot.session.manager import SessionManager

        if self.bus is not None:
            bus = self.bus
        else:
            from nanobot.bus.queue import MessageBus
            bus = MessageBus()
        session_manager = SessionManager(self.workspace)

        cron = None
        if self.enable_cron:
            cron_path = self.workspace / ".cron" / "jobs.json"
            cron_path.parent.mkdir(parents=True, exist_ok=True)
            cron = CronService(cron_path)

        # Temporary shim: AgentLoop still reads send_progress/send_tool_hints
        # from ChannelsConfig. Will be refactored away with AgentLoop.
        channels_cfg = ChannelsConfig(
            send_progress=self.send_progress,
            send_tool_hints=self.send_tool_hints,
        )

        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=self.workspace,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_iterations=self.max_iterations,
            memory_window=self.memory_window,
            reasoning_effort=self.reasoning_effort,
            tools=self.tools,
            cron_service=cron,
            session_manager=session_manager,
            channels_config=channels_cfg,
        )

        if cron is not None:
            cron.on_job = self._make_cron_callback(agent, bus)

        channel_manager = ChannelManager(self.channels, bus)

        heartbeat = None
        if self.enable_heartbeat:
            heartbeat = HeartbeatService(
                workspace=self.workspace,
                provider=self.provider,
                model=agent.model,
                on_execute=self._make_heartbeat_execute(agent, channel_manager, session_manager),
                on_notify=self._make_heartbeat_notify(bus, channel_manager, session_manager),
                interval_s=self.heartbeat_interval_s,
                enabled=True,
            )

        return bus, agent, channel_manager, cron, heartbeat

    @staticmethod
    def _make_cron_callback(agent, bus):
        async def on_cron_job(job):
            from nanobot.agent.tools.cron import CronTool
            from nanobot.agent.tools.message import MessageTool
            from nanobot.bus.events import OutboundMessage

            note = (
                "[Scheduled Task] Timer finished.\n\n"
                f"Task '{job.name}' has been triggered.\n"
                f"Scheduled instruction: {job.payload.message}"
            )

            cron_tool = agent.tools.get("cron")
            token = None
            if isinstance(cron_tool, CronTool):
                token = cron_tool.set_cron_context(True)
            try:
                response = await agent.process_direct(
                    note,
                    session_key=f"cron:{job.id}",
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to or "direct",
                )
            finally:
                if isinstance(cron_tool, CronTool) and token is not None:
                    cron_tool.reset_cron_context(token)

            message_tool = agent.tools.get("message")
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return response

            if job.payload.deliver and job.payload.to and response:
                await bus.publish_outbound(OutboundMessage(
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to,
                    content=response,
                ))
            return response

        return on_cron_job

    @staticmethod
    def _make_heartbeat_execute(agent, channel_manager, session_manager):
        async def on_execute(tasks: str) -> str:
            channel, chat_id = _pick_target(channel_manager, session_manager)

            async def _silent(*_a, **_kw):
                pass

            return await agent.process_direct(
                tasks,
                session_key="heartbeat",
                channel=channel,
                chat_id=chat_id,
                on_progress=_silent,
            )

        return on_execute

    @staticmethod
    def _make_heartbeat_notify(bus, channel_manager, session_manager):
        async def on_notify(response: str) -> None:
            from nanobot.bus.events import OutboundMessage

            channel, chat_id = _pick_target(channel_manager, session_manager)
            if channel == "cli":
                return
            await bus.publish_outbound(OutboundMessage(
                channel=channel, chat_id=chat_id, content=response
            ))

        return on_notify

    async def run(self) -> None:
        """Start all components and run until interrupted."""
        from nanobot.utils.helpers import sync_workspace_templates

        self.workspace.mkdir(parents=True, exist_ok=True)
        sync_workspace_templates(self.workspace)

        bus, agent, channel_manager, cron, heartbeat = self._build()

        logger.info("Nanobot starting (workspace={})", self.workspace)

        try:
            if cron:
                await cron.start()
            if heartbeat:
                await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channel_manager.start_all(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutting down...")
        finally:
            await agent.close_mcp()
            if heartbeat:
                heartbeat.stop()
            if cron:
                cron.stop()
            agent.stop()
            await channel_manager.stop_all()


def _pick_target(channel_manager: ChannelManager, session_manager) -> tuple[str, str]:
    """Pick the best routable channel/chat for proactive delivery."""
    enabled = set(channel_manager.enabled_channels)
    for item in session_manager.list_sessions():
        key = item.get("key") or ""
        if ":" not in key:
            continue
        channel, chat_id = key.split(":", 1)
        if channel in {"cli", "system"}:
            continue
        if channel in enabled and chat_id:
            return channel, chat_id
    return "cli", "direct"
