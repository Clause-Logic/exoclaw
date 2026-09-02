"""Contract tests for opt-in mid-turn user-message steering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import LLMResponse, ToolCallRequest


def _make_loop(*, on_steer: AsyncMock) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    conversation = MagicMock()
    conversation.build_prompt = AsyncMock(
        side_effect=lambda _session_id, message, **_kwargs: [{"role": "user", "content": message}]
    )
    conversation.record = AsyncMock()
    conversation.clear = AsyncMock(return_value=True)
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        conversation=conversation,
        on_steer=on_steer,
    )


class TestSteering:
    async def test_steer_before_tools_skips_every_unstarted_call(self) -> None:
        on_steer = AsyncMock(side_effect=[[], ["Use the local file instead."], [], []])
        loop = _make_loop(on_steer=on_steer)
        tool_response = LLMResponse(
            content="I'll inspect both.",
            tool_calls=[
                ToolCallRequest(id="call-1", name="web_search", arguments={}),
                ToolCallRequest(id="call-2", name="web_fetch", arguments={}),
            ],
        )
        loop.provider.chat = AsyncMock(side_effect=[tool_response, LLMResponse(content="Updated.")])
        loop._invoke_tool = AsyncMock(return_value=("should not run", None))

        response = await loop._process_message(
            InboundMessage(channel="test", sender_id="u", chat_id="chat", content="research this")
        )

        assert response is not None and response.content == "Updated."
        loop._invoke_tool.assert_not_awaited()
        second_prompt = loop.provider.chat.call_args_list[1].kwargs["messages"]
        assert [message["role"] for message in second_prompt[-4:]] == [
            "assistant",
            "tool",
            "tool",
            "user",
        ]
        assert all("Skipped because" in message["content"] for message in second_prompt[-3:-1])
        assert second_prompt[-1]["content"] == "Use the local file instead."

    async def test_steer_after_a_tool_skips_only_remaining_calls(self) -> None:
        on_steer = AsyncMock(side_effect=[[], [], ["Stop after this result."], [], []])
        loop = _make_loop(on_steer=on_steer)
        first = ToolCallRequest(id="call-1", name="read_file", arguments={})
        second = ToolCallRequest(id="call-2", name="write_file", arguments={})
        loop.provider.chat = AsyncMock(
            side_effect=[
                LLMResponse(content="I'll make two changes.", tool_calls=[first, second]),
                LLMResponse(content="Stopped after reading."),
            ]
        )
        loop._invoke_tool = AsyncMock(return_value=("read complete", None))

        response = await loop._process_message(
            InboundMessage(channel="test", sender_id="u", chat_id="chat", content="update it")
        )

        assert response is not None and response.content == "Stopped after reading."
        loop._invoke_tool.assert_awaited_once_with(first)
        second_prompt = loop.provider.chat.call_args_list[1].kwargs["messages"]
        assert [message["content"] for message in second_prompt[-3:]] == [
            "read complete",
            "Skipped because a new user message arrived before this tool started.",
            "Stop after this result.",
        ]

    async def test_steer_after_final_response_reprompts_in_same_turn(self) -> None:
        on_steer = AsyncMock(side_effect=[[], ["Actually, use a shorter answer."], [], []])
        loop = _make_loop(on_steer=on_steer)
        loop.provider.chat = AsyncMock(
            side_effect=[
                LLMResponse(content="A long answer."),
                LLMResponse(content="Short answer."),
            ]
        )

        response = await loop._process_message(
            InboundMessage(channel="test", sender_id="u", chat_id="chat", content="answer me")
        )

        assert response is not None and response.content == "Short answer."
        second_prompt = loop.provider.chat.call_args_list[1].kwargs["messages"]
        assert [message["content"] for message in second_prompt[-2:]] == [
            "A long answer.",
            "Actually, use a shorter answer.",
        ]

    async def test_invalid_or_failed_steering_source_does_not_break_the_turn(self) -> None:
        on_steer = AsyncMock(side_effect=[RuntimeError("store unavailable"), ["not", 1], []])
        loop = _make_loop(on_steer=on_steer)
        loop.provider.chat = AsyncMock(return_value=LLMResponse(content="Completed."))

        response = await loop._process_message(
            InboundMessage(channel="test", sender_id="u", chat_id="chat", content="continue")
        )

        assert response is not None and response.content == "Completed."
        assert loop.provider.chat.await_count == 1
