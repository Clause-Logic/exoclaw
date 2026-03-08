"""Conversation protocol and default implementation."""

from __future__ import annotations

import asyncio
import weakref
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.session.manager import Session, SessionManager


@runtime_checkable
class Conversation(Protocol):
    """
    Structural protocol for conversation state management.

    Covers session storage, memory consolidation, and prompt construction
    in one interface. The default implementation (DefaultConversation) does
    all of this with local files. External packages can replace the whole
    thing — e.g., a Redis-backed store with vector memory.

    External packages implement this without inheriting from any nanobot class:

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
    ) -> list[dict[str, Any]]:
        """Return the full messages list to send to the LLM.

        Includes system prompt, windowed history, and the new user message.
        May trigger async background consolidation if the history is long.
        plugin_context is injected into the system prompt (e.g. from tools/channels
        that implement the optional system_context() hook).
        """
        ...

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
    ) -> None:
        """Persist the messages produced during one turn.

        new_messages should be everything from the user's new message onwards
        (i.e., the user message + all assistant/tool messages added by the loop).
        """
        ...

    async def clear(self, session_id: str) -> bool:
        """Archive current session to memory files and start fresh.

        Returns True on success, False if archival failed (session is not cleared).
        """
        ...

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all known sessions (for routing/heartbeat)."""
        ...


class DefaultConversation:
    """
    File-backed conversation implementation.

    - Sessions: JSONL files in workspace/sessions/
    - Long-term memory: workspace/memory/MEMORY.md
    - History log: workspace/memory/HISTORY.md
    - Prompt: built by ContextBuilder (workspace bootstrap files + memory + skills)

    Memory consolidation runs async in the background when the unconsolidated
    message count exceeds memory_window.
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        workspace: Path,
        provider: Any,  # LLMProvider — avoid circular import
        model: str,
        memory_window: int = 100,
    ):
        self.workspace = workspace
        self._provider = provider
        self._model = model
        self._memory_window = memory_window
        self._sessions = SessionManager(workspace)
        self._context = ContextBuilder(workspace)
        self._memory = MemoryStore(workspace)
        self._consolidating: set[str] = set()
        self._consolidation_tasks: set[asyncio.Task] = set()
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def build_prompt(
        self,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        session = self._sessions.get_or_create(session_id)

        # Kick off background consolidation if history has grown too long.
        unconsolidated = len(session.messages) - session.last_consolidated
        if unconsolidated >= self._memory_window and session_id not in self._consolidating:
            self._consolidating.add(session_id)
            lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._memory.consolidate(
                            session, self._provider, self._model,
                            memory_window=self._memory_window,
                        )
                finally:
                    self._consolidating.discard(session_id)
                    t = asyncio.current_task()
                    if t:
                        self._consolidation_tasks.discard(t)

            task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(task)

        history = session.get_history(max_messages=self._memory_window)
        return self._context.build_messages(
            history=history,
            current_message=message,
            media=media,
            channel=channel,
            chat_id=chat_id,
            plugin_context=plugin_context,
        )

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
    ) -> None:
        from datetime import datetime

        session = self._sessions.get_or_create(session_id)

        for m in new_messages:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")

            # Skip empty assistant messages — they poison the context.
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue

            # Truncate large tool results to keep sessions from bloating.
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"

            # Strip runtime context prefix from user messages before saving.
            elif role == "user":
                tag = ContextBuilder._RUNTIME_CONTEXT_TAG
                if isinstance(content, str) and content.startswith(tag):
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if (
                            c.get("type") == "text"
                            and isinstance(c.get("text"), str)
                            and c["text"].startswith(tag)
                        ):
                            continue
                        if (
                            c.get("type") == "image_url"
                            and c.get("image_url", {}).get("url", "").startswith("data:image/")
                        ):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered

            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)

        session.updated_at = __import__("datetime").datetime.now()
        self._sessions.save(session)

    async def clear(self, session_id: str) -> bool:
        session = self._sessions.get_or_create(session_id)
        lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())
        self._consolidating.add(session_id)
        try:
            async with lock:
                snapshot = session.messages[session.last_consolidated :]
                if snapshot:
                    temp = Session(key=session_id)
                    temp.messages = list(snapshot)
                    ok = await self._memory.consolidate(
                        temp, self._provider, self._model, archive_all=True
                    )
                    if not ok:
                        return False
        except Exception:
            logger.exception("/new archival failed for {}", session_id)
            return False
        finally:
            self._consolidating.discard(session_id)

        session.clear()
        self._sessions.save(session)
        self._sessions.invalidate(session_id)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._sessions.list_sessions()
