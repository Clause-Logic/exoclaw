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
            async def record(self, session_id, new_messages) -> None: ...
            async def clear(self, session_id) -> bool: ...
            def list_sessions(self) -> list[dict]: ...
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

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
    ) -> None:
        """Persist the messages produced during one turn."""
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
