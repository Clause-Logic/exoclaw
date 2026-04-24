"""Executor protocol — routes agent loop I/O through a pluggable execution layer.

The default DirectExecutor calls everything inline. Alternative implementations
can wrap each operation differently (e.g. with different timeout policies,
retry strategies, or execution environments).
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable

from exoclaw.agent.conversation import AppendableConversation, Conversation
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import LLMResponse

if TYPE_CHECKING:
    from exoclaw.agent.loop import AgentLoop


# Source for per-turn prior-history messages.
#
# Invoked by ``DirectExecutor.load_messages`` on each iteration. The
# default implementation (installed by ``set_messages``) closes over an
# immutable list; a disk-backed implementation re-reads a JSONL file
# each call so the list is transient rather than heap-resident for the
# whole turn. See ``docs/memory-model.md`` phase 2b.
PriorSource = Callable[[], list[dict[str, object]]]


def _empty_prior_source() -> list[dict[str, object]]:
    """Default source — returns an empty list. Used before any
    ``set_messages`` / ``set_prior_source`` call runs, so a premature
    ``load_messages`` doesn't raise ``LookupError``.
    """
    return []


def _supports_append(conversation: Conversation) -> TypeGuard[AppendableConversation]:
    """Return True if the Conversation implements ``append`` as a real
    coroutine. Narrows the static type to ``AppendableConversation`` so
    callers can reach the opt-in methods without a cast.

    ``asyncio.iscoroutinefunction`` rather than ``hasattr`` — a
    ``MagicMock`` stand-in has every attribute, and we don't want the
    many tests that patch Conversation with a plain Mock to
    accidentally activate the per-message path and then crash on an
    ``await`` of a non-coroutine.
    """
    fn = getattr(conversation, "append", None)
    return asyncio.iscoroutinefunction(fn)


def _supports_post_turn(conversation: Conversation) -> TypeGuard[AppendableConversation]:
    fn = getattr(conversation, "post_turn", None)
    return asyncio.iscoroutinefunction(fn)


_uuid7_lock = threading.Lock()
_uuid7_last_ms = 0


def _uuid7() -> str:
    """Inline uuidv7 generator.

    Python 3.14 will ship ``uuid.uuid7()`` in stdlib, but exoclaw
    supports 3.11+. This is a minimal RFC 9562-compliant implementation
    — no external dependency, ~20 lines. v7 is time-ordered, so log
    lines sort chronologically when sorted by ``turn.id``.

    The raw ``time.time_ns()`` wall clock can jump backwards under NTP
    adjustments, VM restore, or leap seconds. A naive impl would then
    emit ids whose timestamp prefix regresses, breaking the ordering
    guarantee callers rely on. We clamp against a per-process
    non-decreasing high-water mark inside a small critical section so
    successive calls always see a non-decreasing ms value — even
    during a clock regression, ids stay monotonic at the cost of
    briefly "freezing" the stamped time until wall clock catches up.
    """
    global _uuid7_last_ms
    with _uuid7_lock:
        now_ms = time.time_ns() // 1_000_000
        ts_ms = now_ms if now_ms > _uuid7_last_ms else _uuid7_last_ms
        _uuid7_last_ms = ts_ms

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

    # When True, the executor takes responsibility for publishing the
    # final turn reply to the bus itself and ``_process_message`` returns
    # ``None`` instead of constructing an ``OutboundMessage`` for the
    # caller to publish. Executors opt in when they have a more
    # appropriate place to perform the send than the outer agent loop —
    # the core doesn't care why. When False, ``_process_message`` builds
    # the ``OutboundMessage`` and the caller handles the send.
    handles_response_send: bool

    # When True, the executor takes responsibility for persisting each
    # inbound message the moment a channel hands it over — typically by
    # starting a durable workflow — and ``AgentLoop`` wires
    # ``bus.publish_inbound`` to call ``executor.enqueue_inbound``
    # instead of putting the message on the in-memory asyncio queue.
    # The point is to close the durability gap between "channel received
    # the message" and "agent began processing": with a pass-through
    # executor a container crash in that window loses the message; with
    # a durable executor the message is journaled before the channel's
    # publish call returns.
    #
    # When False (the default), ``publish_inbound`` uses the asyncio
    # queue and ``AgentLoop.run`` drains it, as before.
    handles_inbound_enqueue: bool

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

    async def append_message(
        self,
        conversation: Conversation,
        session_id: str,
        message: dict[str, object],
    ) -> None:
        """Persist a single message mid-turn via the Conversation.

        Durable executors (DBOS, Temporal) wrap this in a step/activity
        so the append is replay-safe on crash recovery. Pass-through
        executors just forward to ``conversation.append``.
        """
        ...

    async def post_turn(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> None:
        """Fire end-of-turn hooks after all messages have been persisted."""
        ...

    async def record(
        self,
        conversation: Conversation,
        session_id: str,
        new_messages: list[dict[str, object]],
    ) -> None:
        """Deprecated — prefer ``append_message`` + ``post_turn``.

        Kept so the agent loop can fall back to end-of-turn persistence
        when a Conversation implementation doesn't support ``append``.
        """
        ...

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
        model: str | None = None,
        publish_response: bool = False,
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

    # NOTE: ``set_prior_source`` lives on ``DirectExecutor`` as a
    # concrete method rather than on this Protocol. Keeping it off
    # the Protocol avoids a breaking static-typing change for
    # external executors (DBOSExecutor and any third-party impl) that
    # previously conformed to ``Executor``. The method is opt-in:
    # impls that want phase 2b's lazy-prior capability add it
    # themselves; callers that rely on it narrow to the concrete
    # executor type (or ``hasattr``-check) at the call site.

    # NOTE: ``enqueue_inbound`` lives on durable executors as a
    # concrete method rather than on this Protocol. Same reasoning
    # as ``set_prior_source`` above — keeping it off the Protocol
    # avoids a breaking static-typing change for external executors
    # that don't opt in. Callers (AgentLoop) guard the call with
    # ``getattr(..., "handles_inbound_enqueue", False)``.

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
    """Pass-through executor — calls everything inline.

    The per-turn message buffer is split into two ContextVars:

    * ``_prior_var`` holds messages the turn inherits from prior
      conversation history (what ``build_prompt`` returns). Set once
      at turn start via ``set_messages`` and never mutated during the
      turn. Compaction replaces it wholesale.
    * ``_delta_var`` holds messages produced during the current turn.
      ``append_messages`` extends it; ``load_messages`` concatenates
      prior + delta for the LLM request body.

    This split is phase 2a of the memory-model doc's three-phase plan
    (see docs/memory-model.md). The structural boundary lets phase 2b
    stop holding prior as a Python list — future versions read prior
    from the session JSONL on demand instead. Phase 2a alone doesn't
    save RAM but establishes the invariant that prior is read-only
    mid-turn and doesn't need to be part of ``append_messages``'
    mutation path.

    Concurrency: both ContextVars are per-executor-instance, per-task.
    A module-level or plain-instance-attr buffer would let two
    concurrent turns trample each other (e.g. a cron firing while a
    user-initiated turn is in flight). asyncio.create_task snapshots
    the current context, so each turn's call chain starts with an
    independent binding on both vars.
    """

    # Pass-through executor does not publish — leaves that to the caller.
    handles_response_send: bool = False

    # Pass-through executor has no durable store to put an inbound
    # message into, so it leaves the bus's asyncio-queue path in place
    # and ``AgentLoop.run`` drains it.
    handles_inbound_enqueue: bool = False

    def __init__(self) -> None:
        # Per-instance ContextVars: each executor has its own bindings,
        # and within a single executor each asyncio task has its own
        # view. Two executors in the same task don't share state.
        #
        # ``_prior_var`` stores a ``PriorSource`` — a callable that
        # returns the prior-history list — rather than the list itself.
        # The ``set_messages`` back-compat path captures the list in
        # a closure (keeps it alive in the Python heap, matching the
        # old behaviour). A new ``set_prior_source`` call site can
        # install a closure that reads from disk on demand, letting
        # prior not be held between LLM iterations. That's phase 2b
        # of the memory-model doc's plan — see docs/memory-model.md.
        self._prior_var: ContextVar[PriorSource] = ContextVar(f"exoclaw_executor_prior_{id(self)}")
        self._delta_var: ContextVar[list[dict[str, object]]] = ContextVar(
            f"exoclaw_executor_delta_{id(self)}"
        )

    def _get_prior_source(self) -> PriorSource:
        try:
            return self._prior_var.get()
        except LookupError:
            self._prior_var.set(_empty_prior_source)
            return _empty_prior_source

    def _get_prior(self) -> list[dict[str, object]]:
        # Invokes the source each call — for the ``set_messages`` path
        # that's a no-op (closure returns the held list); for a
        # ``set_prior_source`` disk-backed path it materialises fresh
        # bytes, which the caller typically drops after a single
        # ``load_messages`` → ``chat`` cycle.
        return self._get_prior_source()()

    def _get_delta(self) -> list[dict[str, object]]:
        try:
            return self._delta_var.get()
        except LookupError:
            buf: list[dict[str, object]] = []
            self._delta_var.set(buf)
            return buf

    def append_messages(self, messages: list[dict[str, object]]) -> None:
        self._get_delta().extend(messages)

    def load_messages(self) -> list[dict[str, object]]:
        # Concat into a new list — callers treat the return as
        # owned (they may pass it to httpx which serialises it, or
        # mutate it in a compaction callback). Must not be a view
        # onto prior + delta that would change under their feet if
        # append_messages runs concurrently.
        return [*self._get_prior(), *self._get_delta()]

    def set_messages(self, messages: list[dict[str, object]]) -> None:
        # Called from two places:
        # 1. ``build_prompt``, which seeds prior at turn start. We
        #    clear delta here too — sequential turns on the same
        #    executor and asyncio task share the ContextVar binding,
        #    so without the clear the next turn would see the prior
        #    turn's delta messages leaked into its load_messages()
        #    return.
        # 2. The compaction path, which replaces both prior and
        #    whatever grew in delta this turn with the compacted
        #    list. So we clear delta to avoid double-counting.
        #
        # Snapshot + closure: matches pre-refactor semantics. A peer
        # task that captured ``load_messages`` while ``set_messages``
        # runs concurrently sees its own snapshot — the snapshot list
        # is never mutated after the closure captures it.
        #
        # No explicit delta clear here — ``set_prior_source`` owns
        # that invariant (both entry points need it, so centralising
        # avoids drift between the two).
        snapshot = list(messages)
        self.set_prior_source(lambda: snapshot)

    def set_prior_source(self, source: PriorSource) -> None:
        """Install a prior-history source. Each ``load_messages`` call
        invokes ``source()`` to materialise the prior list, then
        concatenates with the live delta.

        Callers that can read prior from disk (e.g. ``Conversation``
        impls backed by a JSONL session file) use this to keep prior
        from staying in the Python heap between LLM iterations —
        phase 2b of the memory-model doc's three-phase plan.

        ``set_messages`` is the back-compat wrapper that captures a
        list in a closure; behaviour for existing callers is
        unchanged. New callers should prefer this method when they
        have a cheap disk read available.

        Also clears delta for the same reason ``set_messages`` does:
        sequential turns on the same task share the ContextVar
        binding, and a stale delta from the previous turn would leak
        into the new turn's ``load_messages`` return otherwise.
        """
        self._prior_var.set(source)
        self._delta_var.set([])

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

    async def append_message(
        self,
        conversation: Conversation,
        session_id: str,
        message: dict[str, object],
    ) -> None:
        # Pass-through — DirectExecutor has no workflow replay to worry
        # about. Durable executors override this with a step-wrapped call.
        if _supports_append(conversation):
            await conversation.append(session_id, message)

    async def post_turn(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> None:
        if _supports_post_turn(conversation):
            await conversation.post_turn(session_id)

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
        model: str | None = None,
        publish_response: bool = False,
        **kwargs: list[str] | None,
    ) -> tuple[str | None, list[dict[str, object]]] | None:
        return None
