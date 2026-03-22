"""Executor protocol — routes agent loop I/O through a pluggable execution layer.

The default DirectExecutor calls everything inline. Alternative implementations
can wrap each operation differently (e.g. with different timeout policies,
retry strategies, or execution environments).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import LLMResponse


@runtime_checkable
class Executor(Protocol):
    """Pluggable execution layer for agent loop I/O."""

    async def chat(
        self,
        provider: LLMProvider,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> LLMResponse: ...

    async def execute_tool(
        self,
        registry: ToolRegistry,
        name: str,
        params: dict[str, object],
        ctx: ToolContext | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> str: ...

    async def build_prompt(
        self,
        conversation: Conversation,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        **kwargs: list[str] | None,
    ) -> list[dict[str, object]]: ...

    async def record(
        self,
        conversation: Conversation,
        session_id: str,
        new_messages: list[dict[str, object]],
    ) -> None: ...

    async def clear(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> bool: ...

    async def run_hook(
        self,
        fn: Callable[..., Awaitable[object]],
        /,
        *args: object,
        **kwargs: object,
    ) -> object: ...


class DirectExecutor:
    """Pass-through executor — calls everything inline."""

    async def chat(
        self,
        provider: LLMProvider,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        return await provider.chat(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    async def execute_tool(
        self,
        registry: ToolRegistry,
        name: str,
        params: dict[str, object],
        ctx: ToolContext | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> str:
        return await registry.execute(name, params, ctx)

    async def build_prompt(
        self,
        conversation: Conversation,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        **kwargs: list[str] | None,
    ) -> list[dict[str, object]]:
        return await conversation.build_prompt(
            session_id,
            message,
            channel=channel,
            chat_id=chat_id,
            media=media,
            plugin_context=plugin_context,
            **kwargs,
        )

    async def record(
        self,
        conversation: Conversation,
        session_id: str,
        new_messages: list[dict[str, object]],
    ) -> None:
        await conversation.record(session_id, new_messages)

    async def clear(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> bool:
        return await conversation.clear(session_id)

    async def run_hook(
        self,
        fn: Callable[..., Awaitable[object]],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        return await fn(*args, **kwargs)
