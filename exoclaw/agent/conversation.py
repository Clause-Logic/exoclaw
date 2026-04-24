"""Conversation protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Conversation(Protocol):
    """
    Structural protocol for conversation state management.

    External packages implement this without inheriting from any exoclaw class:

        class MyConversation:
            async def build_prompt(self, session_id, message, **kw) -> list[dict]: ...
            async def append(self, session_id, message) -> None: ...
            async def post_turn(self, session_id) -> None: ...
            async def record(self, session_id, new_messages) -> None: ...  # deprecated
            async def clear(self, session_id) -> bool: ...
            def list_sessions(self) -> list[dict]: ...

    Persistence: implementations should support per-message persistence
    via ``append`` — the agent loop calls it after each assistant response,
    tool result, and the incoming user message. ``record`` is retained for
    backward compatibility with implementations that only persist at
    turn-end, but new implementations should prefer ``append`` + ``post_turn``
    so crash recovery sees work-in-progress state and the in-memory buffer
    doesn't grow for a whole turn before flushing.
    """

    async def build_prompt(
        self,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        **kwargs: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Return the full messages list to send to the LLM."""
        ...

    async def append(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> None:
        """Persist a single message as it is produced during a turn.

        Called after each assistant response, tool result, and the
        incoming user message. Should be idempotent on message identity
        so durable-executor replay doesn't double-write — implementations
        can rely on the agent loop producing message objects in the order
        they were generated and with a stable shape per object.

        Implementations that can't support per-message persistence may
        leave this as a no-op and keep their ``record`` implementation;
        the loop will fall back to end-of-turn ``record`` when no
        ``append`` is wired up.
        """
        ...

    async def post_turn(self, session_id: str) -> None:
        """Fire end-of-turn hooks after all messages have been persisted.

        Called once per turn after the loop finishes. Implementations
        use this for agent_end hooks, consolidation triggers, and
        anything else that needs a "turn complete" signal without a
        per-message granularity.
        """
        ...

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
    ) -> None:
        """Persist the messages produced during one turn.

        Deprecated in favour of ``append`` + ``post_turn``. Retained for
        back-compat with external implementations that haven't migrated.
        The agent loop calls this only when ``append`` is not implemented
        on the Conversation.
        """
        ...

    async def clear(self, session_id: str) -> bool:
        """Archive current session and start fresh. Returns True on success."""
        ...

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all known sessions."""
        ...

    def active_tools(self) -> set[str]:
        """Return the set of optional tool names to surface for the current turn.

        Optional hook — implementations that don't need skill-scoped tool
        activation can omit this method.  The agent loop calls it (if present)
        after build_prompt() so the result reflects the skills resolved for
        the current turn.  Return an empty set to suppress all optional tools.
        """
        return set()
