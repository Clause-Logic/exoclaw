"""Executor protocol — routes agent loop I/O through a pluggable execution layer.

The default DirectExecutor calls everything inline. Alternative implementations
can wrap each operation differently (e.g. with different timeout policies,
retry strategies, or execution environments).
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import LLMResponse

if TYPE_CHECKING:
    from exoclaw.agent.loop import AgentLoop


def _uuid7() -> str:
    """Inline uuidv7 generator.

    Python 3.14 will ship ``uuid.uuid7()`` in stdlib, but exoclaw
    supports 3.11+. This is a minimal RFC 9562-compliant implementation
    — no external dependency, ~20 lines. v7 is time-ordered, so log
    lines sort chronologically when sorted by ``turn.id``.
    """
    ts_ms = time.time_ns() // 1_000_000
    rand = secrets.token_bytes(10)
    b = bytearray(16)
    b[0] = (ts_ms >> 40) & 0xFF
    b[1] = (ts_ms >> 32) & 0xFF
    b[2] = (ts_ms >> 24) & 0xFF
    b[3] = (ts_ms >> 16) & 0xFF
    b[4] = (ts_ms >> 8) & 0xFF
    b[5] = ts_ms & 0xFF
    b[6] = 0x70 | (rand[0] & 0x0F)  # version 7 in high nibble
    b[7] = rand[1]
    b[8] = 0x80 | (rand[2] & 0x3F)  # variant 10
    b[9:16] = rand[3:10]
    hx = b.hex()
    return f"{hx[0:8]}-{hx[8:12]}-{hx[12:16]}-{hx[16:20]}-{hx[20:32]}"


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

    async def run_turn(
        self,
        loop: AgentLoop,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        **kwargs: list[str] | None,
    ) -> tuple[str | None, list[dict[str, object]]] | None: ...

    def append_messages(self, messages: list[dict[str, object]]) -> None:
        """Append messages to the executor's backing store for this turn."""
        ...

    def load_messages(self) -> list[dict[str, object]]:
        """Load the current message list for this turn."""
        ...

    def set_messages(self, messages: list[dict[str, object]]) -> None:
        """Replace the current message list (e.g. after compaction)."""
        ...

    async def mint_turn_id(self) -> str:
        """Produce a replay-safe unique id for one turn.

        Called once at the top of each ``AgentLoop._process_turn_inline``
        call. The returned id is bound into structlog's contextvars as
        ``turn.id`` and every downstream log line inherits it, giving
        callers a single-query way to trace everything that happened
        during the turn.

        Durable executors (DBOS, Temporal) must wrap a non-deterministic
        source in whatever their framework provides for replay-safe
        side-effects so the same id is returned on workflow recovery.
        ``DirectExecutor`` just calls ``_uuid7()`` inline — there's no
        replay boundary to worry about.
        """
        ...


class DirectExecutor:
    """Pass-through executor — calls everything inline."""

    def __init__(self) -> None:
        self._messages: list[dict[str, object]] = []

    def append_messages(self, messages: list[dict[str, object]]) -> None:
        self._messages.extend(messages)

    def load_messages(self) -> list[dict[str, object]]:
        return list(self._messages)

    def set_messages(self, messages: list[dict[str, object]]) -> None:
        self._messages = list(messages)

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
        messages = await conversation.build_prompt(
            session_id,
            message,
            channel=channel,
            chat_id=chat_id,
            media=media,
            plugin_context=plugin_context,
            **kwargs,
        )
        self.set_messages(messages)
        return messages

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

    async def mint_turn_id(self) -> str:
        return _uuid7()

    async def run_turn(
        self,
        loop: AgentLoop,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        **kwargs: list[str] | None,
    ) -> tuple[str | None, list[dict[str, object]]] | None:
        return None
