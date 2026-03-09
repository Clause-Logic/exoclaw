"""Tests for exoclaw/app.py coverage."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw.app import Exoclaw as Nanobot
from exoclaw.bus.queue import MessageBus


def _make_provider():
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return provider


def _make_conversation():
    conv = MagicMock()
    conv.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    conv.record = AsyncMock()
    conv.clear = AsyncMock(return_value=True)
    return conv


class TestNanobotBuild:
    def test_build_with_explicit_bus(self):
        bus = MessageBus()
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
            bus=bus,
        )
        result_bus, agent, channel_manager = app._build()
        assert result_bus is bus

    def test_build_without_bus_creates_message_bus(self):
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
        )
        bus, agent, channel_manager = app._build()
        assert isinstance(bus, MessageBus)

    def test_build_returns_agent_with_correct_model(self):
        provider = _make_provider()
        provider.get_default_model.return_value = "gpt-4"
        app = Nanobot(
            provider=provider,
            conversation=_make_conversation(),
        )
        _, agent, _ = app._build()
        assert agent.model == "gpt-4"

    def test_build_with_explicit_model(self):
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
            model="claude-3",
        )
        _, agent, _ = app._build()
        assert agent.model == "claude-3"

    def test_build_with_channels(self):
        ch = MagicMock()
        ch.name = "slack"
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
            channels=[ch],
        )
        _, _, channel_manager = app._build()
        assert channel_manager.get_channel("slack") is ch

    def test_build_with_no_channels(self):
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
        )
        _, _, channel_manager = app._build()
        assert channel_manager.channels == {}

    def test_build_passes_temperature_and_max_tokens(self):
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
            temperature=0.5,
            max_tokens=2048,
            max_iterations=10,
        )
        _, agent, _ = app._build()
        assert agent.temperature == 0.5
        assert agent.max_tokens == 2048
        assert agent.max_iterations == 10

    def test_build_with_tools(self):
        tool = MagicMock()
        tool.name = "my_tool"
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
            tools=[tool],
        )
        _, agent, _ = app._build()
        assert agent.tools.has("my_tool")


class TestNanobotRun:
    async def test_run_calls_agent_run_and_start_all(self):
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
        )

        agent_run_called = asyncio.Event()
        start_all_called = asyncio.Event()

        async def fake_agent_run():
            agent_run_called.set()
            await asyncio.sleep(10)  # will be cancelled

        async def fake_start_all():
            start_all_called.set()
            await asyncio.sleep(10)  # will be cancelled

        bus, agent, channel_manager = app._build()

        with (
            patch.object(agent, "run", side_effect=fake_agent_run),
            patch.object(channel_manager, "start_all", side_effect=fake_start_all),
            patch.object(channel_manager, "stop_all", new_callable=AsyncMock),
            patch.object(app, "_build", return_value=(bus, agent, channel_manager)),
        ):
            run_task = asyncio.create_task(app.run())
            await asyncio.sleep(0.05)
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

        assert agent_run_called.is_set()
        assert start_all_called.is_set()

    async def test_run_calls_stop_on_shutdown(self):
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
        )

        bus, agent, channel_manager = app._build()

        agent_stop_called = []

        original_stop = agent.stop
        def fake_stop():
            agent_stop_called.append(True)
            original_stop()

        with (
            patch.object(agent, "run", new_callable=AsyncMock),
            patch.object(channel_manager, "start_all", new_callable=AsyncMock),
            patch.object(channel_manager, "stop_all", new_callable=AsyncMock),
            patch.object(agent, "stop", side_effect=fake_stop),
            patch.object(app, "_build", return_value=(bus, agent, channel_manager)),
        ):
            await app.run()

        assert len(agent_stop_called) == 1

    async def test_run_calls_stop_all_on_shutdown(self):
        app = Nanobot(
            provider=_make_provider(),
            conversation=_make_conversation(),
        )

        bus, agent, channel_manager = app._build()

        with (
            patch.object(agent, "run", new_callable=AsyncMock),
            patch.object(channel_manager, "start_all", new_callable=AsyncMock),
            patch.object(channel_manager, "stop_all", new_callable=AsyncMock) as mock_stop_all,
            patch.object(app, "_build", return_value=(bus, agent, channel_manager)),
        ):
            await app.run()

        mock_stop_all.assert_called_once()
