"""Executor protocol — routes agent loop I/O through a pluggable execution layer.

The default DirectExecutor calls everything inline. Alternative implementations
can wrap each operation differently (e.g. with different timeout policies,
retry strategies, or execution environments).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, TypeGuard, runtime_checkable

from exoclaw._compat import (
    IS_MICROPYTHON,
    TaskLocal,
    decode_utf8_lossy,
    iscoroutinefunction,
    make_lock,
    make_scratch_path,
    open_text_writer,
    path_basename,
    random_bytes,
)
from exoclaw.agent.conversation import AppendableConversation, Conversation
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import LLMResponse

if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass

    @dataclass
    class ToolResult:
        """Result of a tool invocation, possibly file-backed.

        Carries either the inline string result or a path to a
        scratch file that holds the full output. Tools that implement
        the ``execute_streaming`` opt-in capability (memory-model.md
        Step D) drain into a scratch file as chunks arrive — the
        executor returns ``ToolResult(content=<short preview>,
        content_file=<path>)`` and the agent loop attaches the path
        to the tool message so a file-backed provider can stream the
        full content into the LLM request body without ever
        materialising it as one Python string.

        Tools without the streaming capability return
        ``ToolResult(content=<full result>, content_file=None)`` —
        the legacy inline path. ``content`` is always populated
        (with a preview when ``content_file`` is set) so callers
        that don't look at ``content_file`` still see something
        diagnostic.

        Real ``@dataclass`` on CPython so downstream serializers
        (``dataclasses.asdict`` in DBOS / nanobot journals) keep
        working unchanged.
        """

        content: str
        content_file: str | None = None

else:  # pragma: no cover (cpython)

    class ToolResult:
        """MicroPython fallback — plain class, see CPython branch
        for the contract. MP strips ``name: type`` annotations at
        compile time so we hand-write ``__init__``."""

        def __init__(self, content: str, content_file: str | None = None) -> None:
            self.content = content
            self.content_file = content_file


if TYPE_CHECKING:
    from exoclaw.agent.loop import AgentLoop


# Source for per-turn prior-history messages.
#
# Invoked by ``DirectExecutor.load_messages`` on each iteration. The
# default implementation (installed by ``set_messages``) closes over an
# immutable list; a disk-backed implementation re-reads a JSONL file
# each call so the list is transient rather than heap-resident for the
# whole turn. See ``docs/memory-model.md`` phase 2b.
#
# ``Callable[..., list[dict[...]]]`` is a runtime subscription
# (type alias, not annotation) — MicroPython 1.27 doesn't support
# ``list[X]`` / ``dict[X, Y]`` parameterization at runtime, only in
# annotations (which ``__future__.annotations`` makes strings). Gate
# the real alias on ``TYPE_CHECKING`` so type checkers see it; at
# runtime ``PriorSource`` is just ``object`` and stringified
# annotations resolve via the type-checker side.
if TYPE_CHECKING:
    PriorSource = Callable[[], list[dict[str, object]]]
else:
    PriorSource = object


def _empty_prior_source() -> list[dict[str, object]]:
    """Default source — returns an empty list. Used before any
    ``set_messages`` / ``set_prior_source`` call runs, so a premature
    ``load_messages`` doesn't raise ``LookupError``.
    """
    return []


def _build_lazy_prior_source(
    *,
    full: list[dict[str, object]],
    history_snapshot: list[dict[str, object]],
    reload_history: Callable[[], list[dict[str, object]]],
) -> PriorSource | None:
    """Construct a disk-backed ``PriorSource`` for a turn's prior history.

    Locates ``history_snapshot`` inside ``full`` as a contiguous sublist
    by DICT-EQUALITY match (not ``id()``). This is the critical detail
    for working against real Conversation impls:
    ``DefaultConversation.session.get_history`` strips timestamps /
    consolidation fields and returns FRESH dict objects per call, so
    call-#1 (from ``build_prompt``) and call-#2 (from
    ``load_persisted_history``) return dicts with the same content but
    different ``id()``.

    Returns ``None`` when no contiguous match is found (empty history,
    a PromptBuilder that transforms messages, isolated mode). The
    caller then falls back to the closure-over-list snapshot path
    (``set_messages``).

    The returned source closes over ``prefix`` and ``suffix`` (small,
    stable for the turn) and invokes ``reload_history`` on each call.
    RAM savings come from the history slice — usually the bulk of
    prompt size — not being heap-resident between LLM iterations.

    Mirrors the helper in ``exoclaw_executor_dbos.executor`` — both
    executors implement the same phase 2b auto-wire so the
    ``streaming_history``-backed Conversation works under either.
    """
    # Defensive type guard: tests routinely pass a bare ``MagicMock``
    # as ``conversation``, in which case ``load_persisted_history``
    # resolves to an auto-mock and ``loader(...)`` returns a Mock —
    # which is truthy, ``len()``-able (returns 0), and indexable
    # (returns more Mocks). Without this check we'd silently install
    # a bogus prior source on every Mock'd test. Real impls return
    # ``list[dict[str, object]]``; refuse anything else.
    if not isinstance(history_snapshot, list):
        return None
    if not history_snapshot:
        return None
    n = len(history_snapshot)
    if n > len(full):
        return None
    # Element-wise compare rather than ``full[i:i+n] == history_snapshot``
    # — slicing allocates a fresh list on every iteration. For a large
    # ``full`` with a large history window, that's a lot of short-lived
    # lists on the hot ``build_prompt`` path.
    first_item = history_snapshot[0]
    first_idx: int | None = None
    for i in range(len(full) - n + 1):
        if full[i] != first_item:
            continue
        for j in range(1, n):
            if full[i + j] != history_snapshot[j]:
                break
        else:
            first_idx = i
            break
    if first_idx is None:
        return None
    last_idx = first_idx + n - 1
    prefix = list(full[:first_idx])
    suffix = list(full[last_idx + 1 :])

    def _source() -> list[dict[str, object]]:
        # ``[*a, *b, *c]`` list-literal unpacking isn't supported on
        # MicroPython 1.27 — concatenate explicitly. ``reload_history``
        # returns a list, so ``+`` works on both runtimes.
        return prefix + reload_history() + suffix

    return _source


def _supports_append(conversation: Conversation) -> TypeGuard[AppendableConversation]:
    """Return True if the Conversation implements ``append`` as a real
    coroutine. Narrows the static type to ``AppendableConversation`` so
    callers can reach the opt-in methods without a cast.

    ``iscoroutinefunction`` rather than ``hasattr`` — a ``MagicMock``
    stand-in has every attribute, and we don't want the many tests
    that patch Conversation with a plain Mock to accidentally
    activate the per-message path and then crash on an ``await`` of
    a non-coroutine.

    Tries the **class** attribute first, then falls back to the
    instance attribute. The two-step lookup handles both runtimes:

    - MicroPython: bound methods have ``__class__.__name__ ==
      "bound_method"`` and don't expose ``__func__`` / ``__self__``,
      so introspecting the bound method returns False even when the
      underlying ``async def`` is what we want. ``type(conv).append``
      returns the unbound function whose class IS ``generator`` for
      ``async def`` — caught by the class-attribute path.
    - CPython with ``MagicMock(spec=...)``: ``type(mock).append`` is
      None (MagicMock doesn't put spec attrs on the type, only the
      instance), and the test sets ``mock.append = AsyncMock(...)``
      on the instance. Falls through to the instance attribute, which
      ``inspect.iscoroutinefunction`` correctly recognises as async.
    """
    fn = getattr(type(conversation), "append", None)
    if fn is None:
        fn = getattr(conversation, "append", None)
    return iscoroutinefunction(fn)


def _supports_post_turn(conversation: Conversation) -> TypeGuard[AppendableConversation]:
    fn = getattr(type(conversation), "post_turn", None)
    if fn is None:
        fn = getattr(conversation, "post_turn", None)
    return iscoroutinefunction(fn)


_uuid7_lock = make_lock()
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

    rand = random_bytes(10)
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
    handles_response_send: bool  # pragma: no cover (micropython)

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
    handles_inbound_enqueue: bool  # pragma: no cover (micropython)

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

    async def execute_tool_with_handle(
        self,
        registry: ToolRegistry,
        name: str,
        params: dict[str, object],
        ctx: ToolContext | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Execute a tool, returning a possibly-file-backed result.

        Default implementation calls :meth:`execute_tool` and wraps
        the string result with no ``content_file`` — preserves the
        legacy inline path for executors that don't override this.
        Step-D-aware executors (``DirectExecutor``, future durable
        executors) override to detect ``Tool.execute_streaming`` and
        drain it into a scratch file.

        Callers (the agent loop) should prefer this method so that
        tools opting into the streaming capability get the
        memory-model Step D win automatically.
        """
        result = await self.execute_tool(registry, name, params, ctx, tool_call_id=tool_call_id)
        return ToolResult(content=result, content_file=None)

    def monotonic_ms(self) -> int:
        """Return a millisecond-resolution monotonic clock value.

        DirectExecutor reads real wall time. Durable executors that run
        agent code inside a sandboxed workflow runtime (e.g. Temporal,
        DBOS) implement this to return their workflow's deterministic
        clock so duration telemetry doesn't trip the sandbox's wall-clock
        restrictions or break replay. The agent loop uses the return only
        for telemetry (``turn.duration_ms`` and similar), so under-reporting
        is acceptable.
        """
        ...

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

    def monotonic_ms(self) -> int:
        """Real wall-clock monotonic time. Durable executors override."""
        from exoclaw._compat import monotonic_ms as _mod_monotonic_ms

        return _mod_monotonic_ms()

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
        self._prior_var: TaskLocal[PriorSource] = TaskLocal(f"exoclaw_executor_prior_{id(self)}")
        self._delta_var: TaskLocal[list[dict[str, object]]] = TaskLocal(
            f"exoclaw_executor_delta_{id(self)}"
        )
        # Per-task list of scratch files written by streaming tool
        # results during this turn. Cleaned up at ``post_turn``.
        # See ``execute_tool_with_handle`` and memory-model.md Step D.
        self._scratch_paths_var: TaskLocal[list[str]] = TaskLocal(
            f"exoclaw_executor_scratch_{id(self)}"
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
        # ``+`` rather than ``[*a, *b]`` because PEP 448 list-literal
        # unpacking isn't supported on MicroPython 1.27.
        return self._get_prior() + self._get_delta()

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

    async def execute_tool_with_handle(
        self,
        registry: ToolRegistry,
        name: str,
        params: dict[str, object],
        ctx: ToolContext | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Step D opt-in: detect ``Tool.execute_streaming`` and drain
        chunks into a scratch file as they arrive, returning a
        file-backed ``ToolResult``. Falls back to the inline
        ``execute_tool`` path for tools that don't opt in.

        Tools that *do* opt in are responsible for yielding chunks
        no smaller than they'd want to commit to disk at once —
        usually whatever line / record granularity their underlying
        producer (subprocess stdout, HTTP body, file read) emits.
        The executor never buffers more than the current chunk in
        memory, so per-chunk size is the upper bound on transient
        Python heap pressure for a single tool call.

        ``execute_with_context`` and the streaming opt-in are
        independent — a tool can have either, neither, or both.
        Streaming wins when both are present (this version doesn't
        pass ``ctx`` to ``execute_streaming``; tools that need
        session context for streaming should add a future
        ``execute_streaming_with_context`` overload).
        """
        resolved = registry.stream_dispatch(name, params)
        if isinstance(resolved, str):
            return ToolResult(content=resolved, content_file=None)
        tool, validated = resolved

        streamer = getattr(tool, "execute_streaming", None)
        if streamer is None or not callable(streamer):
            # Tool doesn't expose ``execute_streaming`` — inline path.
            inline = await self.execute_tool(registry, name, params, ctx, tool_call_id=tool_call_id)
            return ToolResult(content=inline, content_file=None)

        # Per-runtime dispatch.
        #
        # CPython: ``inspect.isasyncgenfunction`` is the strict
        # pre-call discriminator. It rejects ``MagicMock(spec=...)``-
        # derived auto-attrs (a Mock's ``execute_streaming`` slot is
        # callable but isn't an async generator function), so test
        # mocks fall through to the inline path correctly. The
        # streamer result drains via ``async for``.
        #
        # MicroPython: ``async def f(): yield ...`` compiles to a
        # **plain generator** instance — no ``__aiter__`` /
        # ``__anext__``, and uasyncio's frozen ``asyncio`` doesn't
        # implement the async-iterator protocol. The
        # ``execute_streaming`` Tool-protocol contract is the same
        # source-level shape on both runtimes (``async def`` +
        # ``yield``), but on MP we drain via plain ``for``. Tool
        # authors write the same code; the executor adapts at the
        # iteration site. This is what makes memory-model.md Step D
        # active on MP — a fat tool result drains to a scratch file
        # instead of materialising as one Python string.
        if not IS_MICROPYTHON:  # pragma: no cover (micropython)
            import inspect as _inspect

            if not _inspect.isasyncgenfunction(streamer):
                inline = await self.execute_tool(
                    registry, name, params, ctx, tool_call_id=tool_call_id
                )
                return ToolResult(content=inline, content_file=None)
        try:
            streamer_result = streamer(**validated)
        except TypeError as e:
            # Cross-runtime safety net for arity / param mismatches —
            # an ``async def execute_streaming(self, foo): yield ...``
            # called with kwargs that don't include ``foo`` raises
            # ``TypeError`` on both runtimes. CPython's pre-call
            # ``isasyncgenfunction`` doesn't validate the signature,
            # only the function shape, so this branch is reachable
            # on both. Surface as the same ``Error: ...`` shape
            # ``ToolRegistry.execute`` returns on validation failure.
            return ToolResult(
                content=f"Error executing {name}: {e}",
                content_file=None,
            )

        # Streaming path. ``tool_call_id`` originates outside the
        # executor (LLM / provider) and may contain path separators
        # or other unusual bytes. Sanitize aggressively to ASCII
        # alnum + ``-`` / ``_`` and cap at 64 chars before letting
        # it near a filesystem path.
        suffix = ""
        if tool_call_id:
            # MP's ``str`` doesn't ship ``isalnum``; combine
            # ``isalpha()`` + ``isdigit()`` for the same result on
            # both runtimes.
            safe = "".join(
                c if c.isalpha() or c.isdigit() or c in {"-", "_"} else "_" for c in tool_call_id
            )[:64]
            if safe:
                suffix = f"-{safe}"

        # Drain the async iterator into a per-turn scratch file.
        # The file persists until ``post_turn`` cleans it up; the
        # provider reads from it during request-body assembly.
        path = make_scratch_path(prefix="exoclaw-tool-", suffix=f"{suffix}.txt")
        bytes_written = 0
        # Preview budget is enforced **in bytes**, not characters —
        # ``bytes_written`` accounts in bytes, so mixing the two would
        # let non-ASCII output exceed the cap. ``newline=""`` prevents
        # Windows-style ``\n``→``\r\n`` translation that would also
        # desync byte counts vs. on-disk size.
        preview_chunks: list[bytes] = []
        preview_budget = 256
        try:
            with open_text_writer(path) as fh:
                # Bind ``self`` (the registry) into the dispatch
                # ContextVar for the duration of the stream — fan-out
                # tools that call ``ToolRegistry.current()`` from
                # inside ``execute_streaming`` need this exactly the
                # same way the inline ``execute`` path provides it.
                with registry.bind_dispatch():
                    if not IS_MICROPYTHON:  # pragma: no cover (micropython)
                        async for chunk in streamer_result:
                            if not isinstance(chunk, str):
                                chunk = str(chunk)
                            chunk_bytes = chunk.encode("utf-8")
                            fh.write(chunk)
                            bytes_written += len(chunk_bytes)
                            if preview_budget > 0:
                                take = min(preview_budget, len(chunk_bytes))
                                preview_chunks.append(chunk_bytes[:take])
                                preview_budget -= take
                    else:  # pragma: no cover (cpython)
                        # On MP, ``async def f(): yield ...`` produces a
                        # plain generator — sync iteration drains it.
                        # The Tool-protocol surface (``async def +
                        # yield``) is the same source-level shape on
                        # both runtimes; what changes is the executor's
                        # iteration site.
                        for chunk in streamer_result:
                            if not isinstance(chunk, str):
                                chunk = str(chunk)
                            chunk_bytes = chunk.encode("utf-8")
                            fh.write(chunk)
                            bytes_written += len(chunk_bytes)
                            if preview_budget > 0:
                                take = min(preview_budget, len(chunk_bytes))
                                preview_chunks.append(chunk_bytes[:take])
                                preview_budget -= take
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            raise

        # Track for cleanup at ``post_turn``.
        try:
            scratch_paths = self._scratch_paths_var.get()
        except LookupError:
            scratch_paths = []
            self._scratch_paths_var.set(scratch_paths)
        scratch_paths.append(path)

        # Reassemble preview as text. ``decode_utf8_lossy`` covers the
        # one edge case where the byte budget cut a multi-byte UTF-8
        # codepoint mid-sequence — the resulting partial bytes are
        # dropped from the preview rather than crashing the decode
        # (MP's ``bytes.decode`` doesn't accept ``errors="ignore"``).
        preview = decode_utf8_lossy(b"".join(preview_chunks))
        if bytes_written > len(preview.encode("utf-8")):
            preview = f"{preview}…\n[streamed {bytes_written} bytes to {path_basename(path)}]"
        return ToolResult(content=preview, content_file=path)

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
        # Phase 2b auto-wire: when the Conversation exposes a sync
        # ``load_persisted_history(session_id)``, install a disk-backed
        # ``PriorSource`` so the history slice is re-read per
        # ``load_messages`` call instead of being heap-resident in
        # ``_prior_var`` for the whole turn (and bleeding into the
        # *next* turn until ``set_messages`` runs again).
        #
        # Falls back to ``set_messages`` when:
        #   * the conversation doesn't expose ``load_persisted_history``,
        #   * the loader returns an empty history (nothing to be lazy
        #     about),
        #   * ``_build_lazy_prior_source`` can't locate the history
        #     slice in ``messages`` (a PromptBuilder transformed it,
        #     isolated mode dropped it, etc.).
        # See ``docs/memory-model.md`` phase 2b / Step C.
        loader = getattr(conversation, "load_persisted_history", None)
        # Two-step lookup so MP's class-attribute path catches async
        # methods (bound methods on MP aren't introspectable) and
        # CPython's instance-attribute path catches mocks. See
        # ``_supports_append`` above for the full rationale.
        loader_fn = getattr(type(conversation), "load_persisted_history", None)
        if loader_fn is None:
            loader_fn = loader
        if callable(loader) and not iscoroutinefunction(loader_fn):
            history_snapshot = loader(session_id)
            source = _build_lazy_prior_source(
                full=messages,
                history_snapshot=history_snapshot,
                reload_history=lambda: loader(session_id),
            )
            if source is not None:
                self.set_prior_source(source)
                return messages
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
        # Clean up any tool-result scratch files written during this
        # turn. Holding them past the turn would leak — the provider
        # has already serialised their content into the LLM request
        # body, so by the time post_turn runs they're no longer
        # referenced. Best-effort: a missing file is fine (manual
        # cleanup, OS tmpwatch); other OSErrors get swallowed because
        # we don't want a stat-style hiccup to break end-of-turn hooks.
        try:
            scratch_paths = self._scratch_paths_var.get()
        except LookupError:
            scratch_paths = []
        for path in scratch_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        if scratch_paths:
            self._scratch_paths_var.set([])

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
