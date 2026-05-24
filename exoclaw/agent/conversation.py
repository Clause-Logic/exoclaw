"""Conversation protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from exoclaw.agent.hooks import HookRegistration


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

    Implementations that can persist incrementally should also implement
    ``AppendableConversation`` — the agent loop detects that at runtime
    and switches to per-message flushing (see below).
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
        """Persist the messages produced during one turn.

        The agent loop calls this at end-of-turn as the default
        persistence path. Implementations that support per-message
        persistence should additionally implement
        ``AppendableConversation`` — the loop will flush each message
        as it's produced and skip the end-of-turn ``record`` call.
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

        Optional hook — implementations that don't need per-turn tool
        activation can omit this method.  The agent loop calls it (if present)
        after build_prompt() so the result reflects whatever the conversation
        resolved for the current turn.  Return an empty set to suppress all
        optional tools.
        """
        return set()

    def active_hooks(self, event: str) -> list[HookRegistration]:
        """Return the lifecycle hooks active for the current turn at ``event``.

        Optional hook — sibling to ``active_tools``. The agent loop calls it
        at each lifecycle seam (``before_tool``, ``before_finish``) and runs
        the returned handlers in priority order. A Conversation that opts in
        returns whatever hooks it considers active this turn, so behaviour can
        be made conditional — but core stays agnostic, it only sees
        ``(handler, priority)`` pairs. Return an empty list to fire nothing.
        """
        return []

    def run_context(self) -> dict[str, Any]:
        """Return the per-run context bag exposed to hooks via
        ``HookContext.run_context``.

        Optional. Lets the host surface authoritative per-run values (e.g. a
        cycle id minted before the turn) so a hook can read them instead of
        trusting the model's tool args. Defaults to empty.
        """
        return {}

    # NOTE: ``recover_from_overflow`` lives on concrete Conversation
    # implementations as an optional method rather than on this Protocol.
    # Same reasoning as ``Executor.set_prior_source`` and
    # ``Executor.enqueue_inbound`` — keeping it off the Protocol avoids a
    # breaking static-typing change for external Conversation impls that
    # already conform structurally. When the provider raises
    # ``ContextWindowExceededError``, the agent loop calls
    # ``executor.recover_from_overflow(conversation, session_id)``;
    # the executor's default implementation in turn forwards to
    # ``getattr(conversation, "recover_from_overflow", None)``. Implementations
    # that opt in should provide:
    #
    #     async def recover_from_overflow(
    #         self, session_id: str
    #     ) -> list[dict[str, object]] | None: ...
    #
    # Returning a new message list signals the loop to ``set_messages`` and
    # retry the failed iteration; returning ``None`` signals "couldn't
    # recover, surface the original error".


@runtime_checkable
class AppendableConversation(Conversation, Protocol):
    """Opt-in extension: implementations that can persist one message
    at a time as the turn progresses.

    When the agent loop sees a Conversation that satisfies this
    protocol, it calls ``append`` after each new assistant response,
    tool result, and the incoming user message — keeping crash recovery
    from losing mid-turn work and keeping the in-memory message buffer
    from being the sole holder of turn state. ``post_turn`` then runs
    end-of-turn hooks; ``record`` is skipped entirely on this path
    (it would double-write messages already on disk).

    Do NOT implement ``append`` as an async no-op just to satisfy the
    type — the loop's capability check sees the presence of ``append``
    and skips ``record``, so a no-op ``append`` would drop persistence
    entirely. If your backing store can't persist per-message, keep
    your implementation on the base ``Conversation`` protocol and let
    the loop use the ``record`` fallback.
    """

    async def append(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> None:
        """Persist a single message as it is produced during a turn.

        Called after each assistant response, tool result, and the
        incoming user message. Should be idempotent on message identity
        so durable-executor replay doesn't double-write — implementations
        can rely on the agent loop producing message objects in the
        order they were generated and with a stable shape per object.
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
