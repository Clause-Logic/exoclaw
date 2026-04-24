"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import cast

import structlog
import structlog.contextvars
from structlog.typing import FilteringBoundLogger

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import Tool, ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.bus.events import InboundMessage, OutboundMessage
from exoclaw.bus.protocol import Bus
from exoclaw.executor import DirectExecutor, Executor
from exoclaw.iteration_policy import IterationPolicy
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import ContextWindowExceededError, ToolCallRequest

_UNSET: object = object()  # sentinel: distinguishes "not provided" from explicit None


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Asks the Conversation for the prompt
    3. Calls the LLM
    4. Executes tool calls
    5. Records the turn and sends the response back
    """

    def __init__(
        self,
        bus: Bus,
        provider: LLMProvider,
        conversation: Conversation,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        tools: list[Tool] | None = None,
        registry: ToolRegistry | None = None,
        # Optional lifecycle callbacks — all default to None (backwards compatible).
        # Plugins pass these in; the loop calls them at the appropriate point.
        on_pre_context: Callable[[str, str, str, str], Awaitable[str]] | None = None,
        on_pre_tool: Callable[[str, dict[str, object], str], Awaitable[str | None]] | None = None,
        on_post_turn: Callable[[list[dict[str, object]], str, str, str], Awaitable[None]]
        | None = None,
        on_max_iterations: Callable[[str, str, str], Awaitable[None]] | None = None,
        # Called with all tool calls before execution begins (structured, for UI/observability).
        on_tool_calls: Callable[["list[ToolCallRequest]"], Awaitable[None]] | None = None,
        # Called after each tool result (tool_call, result) — for streaming previews.
        on_tool_result: Callable[["ToolCallRequest", str], Awaitable[None]] | None = None,
        # Called when the provider raises ContextWindowExceededError. Receives the current
        # message list and should return a compacted version, or None to give up.
        on_context_overflow: Callable[
            ["list[dict[str, object]]"], Awaitable["list[dict[str, object]] | None"]
        ]
        | None = None,
        iteration_policy: IterationPolicy | None = None,
        executor: Executor | None = None,
        logger: FilteringBoundLogger | None = None,
    ) -> None:
        self.bus = bus
        self._executor: Executor = executor or DirectExecutor()
        self._iteration_policy = iteration_policy
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._extra_tools = tools or []
        self._on_pre_context = on_pre_context
        self._on_pre_tool = on_pre_tool
        self._on_post_turn = on_post_turn
        self._on_max_iterations = on_max_iterations
        self._on_tool_calls = on_tool_calls
        self._on_tool_result = on_tool_result
        self._on_context_overflow = on_context_overflow

        self.conversation = conversation

        self.tools = registry or ToolRegistry()
        for tool in self._extra_tools:
            self.tools.register(tool)
            if hasattr(tool, "set_bus"):
                tool.set_bus(self.bus)  # type: ignore[call-non-callable]
            if hasattr(tool, "set_registry"):
                tool.set_registry(self.tools)  # type: ignore[call-non-callable]
        self._log: FilteringBoundLogger = logger or structlog.get_logger()
        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task[None]]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._current_ctx: ToolContext | None = None  # set before each _run_agent_loop call

    def _notify_tools_inbound(self, msg: InboundMessage) -> None:
        """Let tools that care about inbound messages update their state."""
        for tool in self.tools._tools.values():
            if hasattr(tool, "on_inbound"):
                tool.on_inbound(msg)  # type: ignore[call-non-callable]

    def _collect_plugin_context(self) -> list[str]:
        """Collect system_context() strings from tools that provide them."""
        ctx = []
        for tool in self.tools._tools.values():
            if hasattr(tool, "system_context"):
                try:
                    result = tool.system_context()  # type: ignore[call-non-callable]
                    if result and isinstance(result, str):
                        ctx.append(result)
                except Exception:
                    self._log.exception(
                        "system_context_error", **{"tool.name": getattr(tool, "name", "?")}
                    )
        return ctx

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list[ToolCallRequest]) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc: ToolCallRequest) -> str:
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return str(tc.name)
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _should_continue(self, iteration: int, tools_used: list[str]) -> bool:
        """Check whether the loop should keep iterating.

        If an ``IterationPolicy`` was provided, delegate to it.
        Otherwise fall back to the hard ``max_iterations`` cap.
        """
        if self._iteration_policy is not None:
            return await self._iteration_policy.should_continue(iteration, tools_used)
        return iteration < self.max_iterations

    async def _build_limit_message(self, iteration: int, tools_used: list[str]) -> str:
        """Build the message shown when the iteration limit is reached.

        If an ``IterationPolicy`` was provided, delegate to it.
        Otherwise return a default message.
        """
        if self._iteration_policy is not None:
            return await self._iteration_policy.on_limit_reached(iteration, tools_used)
        return (
            f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
            "without completing the task. You can try breaking the task into smaller steps."
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict[str, object]],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        model: str | None = None,
        *,
        session_id: str | None = None,
    ) -> tuple[str | None, list[str], list[dict[str, object]]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages).

        When the Conversation supports ``append`` and ``session_id`` is
        provided, each new assistant message and tool result is flushed
        to disk as it's produced — keeps crash recovery from losing
        mid-turn work, and keeps the in-memory buffer from having to
        be the sole holder of turn state.

        ``session_id`` is keyword-only so test stubs that replace this
        method with a narrower signature continue to work — callers
        inside exoclaw always pass it; external stubs keep their
        existing (initial_messages, on_progress, model) shape.

        Does NOT seed the executor via
        ``set_messages(initial_messages)`` — that would overwrite the
        phase-2b ``PriorSource`` that ``build_prompt`` just installed,
        defeating the RAM reduction. In the production flow
        (``_process_turn_inline`` → ``build_prompt`` → this method)
        the executor is already seeded.

        ``initial_messages`` is retained on the signature for
        backwards compatibility with test shims and any external
        caller that monkey-patches this method, but it is NOT used
        here. Callers that invoke ``_run_agent_loop`` directly
        without going through ``build_prompt`` (typically tests)
        must seed the executor themselves if they care about
        message content reaching the provider.

        See ``docs/memory-model.md`` "Step A" for the history.
        """
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        effective_model = model or self.model
        from exoclaw.executor import _supports_append as _has_append

        # Flushing decision is fully a function of what was passed in
        # — no reliance on ``self._current_ctx``, which might be stale
        # from a prior turn or unset on paths that skip _process_message
        # (e.g. system-message dispatch). Flagged by PR #26 review.
        prefer_append = session_id is not None and _has_append(self.conversation)

        async def _flush(msg: dict[str, object]) -> None:
            # ``prefer_append`` implies ``session_id is not None``;
            # the explicit ``is not None`` repeat lets the static type
            # checker narrow ``session_id`` to ``str`` for the call.
            if prefer_append and session_id is not None:
                await self._executor.append_message(self.conversation, session_id, msg)

        while await self._should_continue(iteration, tools_used):
            iteration += 1

            _include = (
                self.conversation.active_tools()
                if hasattr(self.conversation, "active_tools")
                else None
            )
            messages = self._executor.load_messages()
            try:
                response = await self._executor.chat(
                    self.provider,
                    messages=messages,
                    tools=self.tools.get_definitions(include=_include),
                    model=effective_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
            except ContextWindowExceededError:
                if self._on_context_overflow:
                    compacted = await self._on_context_overflow(messages)
                    if compacted is not None:
                        self._log.info("context_compact", iteration=iteration)
                        self._executor.set_messages(compacted)
                        continue
                self._log.error("context_overflow", iteration=iteration)
                final_content = (
                    "The conversation exceeded the model's context window "
                    "and I couldn't recover. Try starting a new session."
                )
                break

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)
                if self._on_tool_calls:
                    await self._executor.run_hook(self._on_tool_calls, response.tool_calls)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
                msg: dict[str, object] = {"role": "assistant", "content": response.content}
                if tool_call_dicts:
                    msg["tool_calls"] = tool_call_dicts
                if response.reasoning_content is not None:
                    msg["reasoning_content"] = response.reasoning_content
                if response.thinking_blocks:
                    msg["thinking_blocks"] = response.thinking_blocks
                self._executor.append_messages([msg])
                await _flush(msg)

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    # `tool_call` + `tool_result` form a start/stop pair
                    # correlated by `tool.call_id`. On error, `.exception()`
                    # attaches the traceback via structlog's format_exc_info.
                    self._log.info(
                        "tool_call",
                        **{
                            "tool.name": tool_call.name,
                            "tool.call_id": tool_call.id,
                        },
                        args=args_str[:200],
                    )
                    t0 = time.monotonic()
                    status = "ok"
                    exc: BaseException | None = None
                    result = ""
                    try:
                        if self._on_pre_tool:
                            sk = self._current_ctx.session_key if self._current_ctx else ""
                            rejection = await self._executor.run_hook(
                                self._on_pre_tool,
                                tool_call.name,
                                tool_call.arguments,
                                sk,
                            )
                            if rejection:
                                status = "rejected"
                                self._log.info(
                                    "tool_reject",
                                    **{
                                        "tool.name": tool_call.name,
                                        "tool.call_id": tool_call.id,
                                    },
                                    reason=str(rejection)[:100],
                                )
                                result = str(rejection)
                            else:
                                result = await self._executor.execute_tool(
                                    self.tools,
                                    tool_call.name,
                                    tool_call.arguments,
                                    self._current_ctx,
                                    tool_call_id=tool_call.id,
                                )
                        else:
                            result = await self._executor.execute_tool(
                                self.tools,
                                tool_call.name,
                                tool_call.arguments,
                                self._current_ctx,
                                tool_call_id=tool_call.id,
                            )
                    except Exception as e:
                        exc = e
                        status = "error"
                        # str(e) can be empty for bare exceptions (e.g. a
                        # no-arg TypeError); fall back to the class name so
                        # the LLM always sees something diagnostic.
                        detail = str(e) or type(e).__name__
                        result = (
                            f"Error executing {tool_call.name}: {detail}"
                            "\n\n[Analyze the error above and try a different approach.]"
                        )
                    finally:
                        duration_ms = int((time.monotonic() - t0) * 1000)
                        stop_kwargs: dict[str, object] = {
                            "tool.name": tool_call.name,
                            "tool.call_id": tool_call.id,
                            "tool.status": status,
                            "tool.duration_ms": duration_ms,
                        }
                        if exc is not None:
                            # Pass exc_info explicitly — by the time this
                            # finally runs, sys.exc_info() has been cleared
                            # by the except handler, so .exception() would
                            # drop the traceback.
                            self._log.error("tool_result", **stop_kwargs, exc_info=exc)
                        else:
                            self._log.info("tool_result", **stop_kwargs)
                    if self._on_tool_result:
                        await self._executor.run_hook(self._on_tool_result, tool_call, result)
                    tool_msg: dict[str, object] = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    }
                    self._executor.append_messages([tool_msg])
                    await _flush(tool_msg)
            else:
                clean = self._strip_think(response.content)
                if response.finish_reason == "error":
                    self._log.error("llm_error", **{"error.message": (clean or "")[:200]})
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                msg2: dict[str, object] = {"role": "assistant", "content": clean}
                if response.reasoning_content is not None:
                    msg2["reasoning_content"] = response.reasoning_content
                if response.thinking_blocks:
                    msg2["thinking_blocks"] = response.thinking_blocks
                self._executor.append_messages([msg2])
                await _flush(msg2)
                final_content = clean
                break

        if final_content is None and not await self._should_continue(iteration, tools_used):
            self._log.warning("iteration_limit", **{"iteration.max": self.max_iterations})
            final_content = await self._build_limit_message(iteration, tools_used)
            if self._on_max_iterations and self._current_ctx:
                ctx = self._current_ctx
                asyncio.ensure_future(
                    self._on_max_iterations(ctx.session_key, ctx.channel, ctx.chat_id)
                )

        return final_content, tools_used, self._executor.load_messages()

    async def process_turn(
        self,
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
    ) -> tuple[str | None, list[dict[str, object]]]:
        """Execute a single turn: build prompt, run agent loop, record.

        If the executor provides a ``run_turn`` method the call is delegated
        to it, allowing durable executors (Temporal, DBOS) to wrap the turn
        in a workflow for crash recovery. Otherwise the turn runs inline.

        ``model`` overrides ``self.model`` for this turn only; falls back to
        the loop default when ``None``.

        Returns ``(final_content, new_messages)``.
        """
        result = await self._executor.run_turn(
            self,
            session_id,
            message,
            channel=channel,
            chat_id=chat_id,
            media=media,
            plugin_context=plugin_context,
            on_progress=on_progress,
            model=model,
            publish_response=publish_response,
            **kwargs,
        )
        if result is not None:
            return result
        return await self._process_turn_inline(
            session_id,
            message,
            channel=channel,
            chat_id=chat_id,
            media=media,
            plugin_context=plugin_context,
            on_progress=on_progress,
            model=model,
            **kwargs,
        )

    async def _process_turn_inline(
        self,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        model: str | None = None,
        **kwargs: list[str] | None,
    ) -> tuple[str | None, list[dict[str, object]]]:
        """The actual turn logic — called directly or from a durable wrapper.

        Binds a per-turn trace context into structlog contextvars so every
        log line emitted during the turn (LLM requests, tool calls, tool
        results, subagent spawns, etc.) carries ``turn.id``,
        ``turn.root_id``, ``turn.parent_id``, ``turn.depth`` and
        ``turn.chain``. That lets a single LogsQL query surface the full
        causal tree of a user message without joins or timestamp
        correlation.

        The fields are read from existing contextvars before binding, so
        nested ``_process_turn_inline`` calls (subagent chains that
        inherit the parent's chain via a workflow argument and re-bind
        before calling ``process_direct``) extend the ancestry correctly.
        """
        turn_id = await self._executor.mint_turn_id()
        ctx = structlog.contextvars.get_contextvars()
        parent_chain = cast(str, ctx.get("turn.chain", "") or "")
        parent_id = ctx.get("turn.id")
        root_id = ctx.get("turn.root_id") or turn_id
        chain = f"{parent_chain}:{turn_id}" if parent_chain else turn_id
        depth = parent_chain.count(":") + 1 if parent_chain else 0

        # Save any existing ``turn.*`` values so a nested call can restore
        # the outer turn's context instead of unbinding it entirely —
        # otherwise the outer turn's subsequent logs (including
        # ``turn_end``) would fire with no trace context bound.
        _turn_keys = (
            "turn.id",
            "turn.root_id",
            "turn.parent_id",
            "turn.depth",
            "turn.chain",
        )
        _sentinel = object()
        prior_turn_ctx = {k: ctx.get(k, _sentinel) for k in _turn_keys}

        structlog.contextvars.bind_contextvars(
            **{
                "turn.id": turn_id,
                "turn.root_id": root_id,
                "turn.parent_id": parent_id,
                "turn.depth": depth,
                "turn.chain": chain,
            }
        )
        turn_start = time.monotonic()
        self._log.info("turn_start")
        try:
            initial = await self._executor.build_prompt(
                self.conversation,
                session_id,
                message,
                channel=channel,
                chat_id=chat_id,
                media=media,
                plugin_context=plugin_context,
                **kwargs,
            )
            # Per-message persistence path. When the Conversation impl
            # supports ``append``, the loop flushes each message as it's
            # produced (see _run_agent_loop); the final post_turn fires
            # end-of-turn hooks. Falls back to the legacy batched
            # ``record`` call for implementations that don't support
            # append.
            from exoclaw.executor import _supports_append as _has_append

            prefer_append = _has_append(self.conversation)
            # Capture everything we need from ``initial`` up front so
            # the frame can release it before the loop iterations run.
            # The executor already holds prior via its ``PriorSource``
            # (phase 2b) — retaining ``initial`` here would duplicate
            # that content in the frame for the lifetime of the turn
            # (which can be minutes while tool calls run). Capture
            # then drop so long-running turns don't keep a copy of
            # the full prior history resident.
            user_msg: dict[str, object] | None = initial[-1] if initial else None
            initial_len = len(initial)
            del initial
            if prefer_append and user_msg is not None:
                # Persist the new user message before the loop runs so a
                # crash mid-turn still has the user's input on disk.
                await self._executor.append_message(self.conversation, session_id, user_msg)
            final_content, _, all_msgs = await self._run_agent_loop(
                [], on_progress=on_progress, model=model, session_id=session_id
            )
            new_msgs = all_msgs[initial_len - 1 :]
            if prefer_append:
                await self._executor.post_turn(self.conversation, session_id)
            else:
                await self._executor.record(self.conversation, session_id, new_msgs)
            return final_content, new_msgs
        finally:
            self._log.info(
                "turn_end",
                **{"turn.duration_ms": int((time.monotonic() - turn_start) * 1000)},
            )
            to_rebind = {k: v for k, v in prior_turn_ctx.items() if v is not _sentinel}
            to_unbind = tuple(k for k, v in prior_turn_ctx.items() if v is _sentinel)
            if to_unbind:
                structlog.contextvars.unbind_contextvars(*to_unbind)
            if to_rebind:
                structlog.contextvars.bind_contextvars(**to_rebind)

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        self._log.info("agent_loop_start")

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if msg.content.strip().lower() == "/stop":
                    await self._handle_stop(msg)
                else:
                    task = asyncio.create_task(self._dispatch(msg))
                    self._active_tasks.setdefault(msg.session_key, []).append(task)

                    def _remove_task(t: asyncio.Task[None], k: str = msg.session_key) -> None:
                        tasks = self._active_tasks.get(k, [])
                        if t in tasks:
                            tasks.remove(t)

                    task.add_done_callback(_remove_task)
        finally:
            all_tasks = [t for ts in self._active_tasks.values() for t in ts if not t.done()]
            for t in all_tasks:
                t.cancel()
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            self._log.info("agent_loop_stop")

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                self._log.exception("task_cancel_error", **{"session.key": msg.session_key})
        sub_cancelled = 0
        for tool in self.tools._tools.values():
            if hasattr(tool, "cancel_by_session"):
                sub_cancelled += await tool.cancel_by_session(msg.session_key)  # type: ignore[call-non-callable]
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
            )
        )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                # Ask ``_process_message`` to let the executor own the
                # publish if it advertises ``handles_response_send``.
                # When it does, we get ``None`` back and skip the publish
                # below; otherwise we publish the returned message as usual.
                response = await self._process_message(msg, publish_response=True)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="",
                            metadata=msg.metadata or {},
                        )
                    )
            except asyncio.CancelledError:
                self._log.info("task_cancel", **{"session.key": msg.session_key})
                raise
            except Exception:
                self._log.exception("message_error", **{"session.key": msg.session_key})
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    )
                )

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None | object = _UNSET,
        model: str | None = None,
        publish_response: bool = False,
        **kwargs: list[str] | None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response.

        ``publish_response`` is forwarded to ``process_turn`` so executors
        that advertise ``handles_response_send`` can take ownership of the
        send. When both flags are True this method returns ``None`` —
        callers read that as "reply already dispatched, nothing more to
        do." Callers that need the reply content back (``process_direct``
        for CLI, subagent chains, cron) pass ``publish_response=False``.
        """
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            sid = msg.session_key_override or f"{channel}:{chat_id}"
            # Rebind only the session-level fields we own — do NOT
            # clear_contextvars. A caller several frames up may have
            # seeded a ``turn.*`` trace context (e.g.
            # ``SubagentManager._run`` threading parent ancestry into
            # a child ``process_direct`` call) and wiping the whole
            # contextvar store would break cross-spawn trace
            # correlation without anyone noticing — the subagent's
            # ``_process_turn_inline`` would see an empty context and
            # start a fresh root. Rebinding the four keys here is
            # idempotent and leaves other bindings alone.
            structlog.contextvars.bind_contextvars(
                **{
                    "session.key": sid,
                    "channel": channel,
                    "chat.id": chat_id,
                    "sender.id": msg.sender_id,
                }
            )
            self._log.info("system_message")
            plugin_ctx = self._collect_plugin_context()
            final_content, _ = await self.process_turn(
                sid,
                msg.content,
                channel=channel,
                chat_id=chat_id,
                plugin_context=plugin_ctx or None,
                model=model or msg.model_override,
                publish_response=publish_response,
            )
            if publish_response and getattr(self._executor, "handles_response_send", False):
                # Executor took ownership of the send — nothing to return.
                return None
            sys_meta = dict(msg.metadata or {})
            sys_meta.setdefault("session_key", sid)
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
                metadata=sys_meta,
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        sid = session_key or msg.session_key

        # Rebind session-level fields only — see the comment above in
        # the system-message branch. Preserving outer ``turn.*``
        # contextvars is what makes subagent trace ancestry survive
        # from ``SubagentManager._run`` into the child's own
        # ``_process_turn_inline``.
        structlog.contextvars.bind_contextvars(
            **{
                "session.key": sid,
                "channel": msg.channel,
                "chat.id": msg.chat_id,
                "sender.id": msg.sender_id,
            }
        )

        self._log.info("message_receive", preview=preview)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            success = await self._executor.clear(self.conversation, sid)
            content = (
                "New session started."
                if success
                else ("Memory archival failed, session not cleared. Please try again.")
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if cmd == "/help":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="🦀 exoclaw commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands",
            )

        self._notify_tools_inbound(msg)
        self._current_ctx = ToolContext(
            session_key=sid,
            channel=msg.channel,
            chat_id=msg.chat_id,
            executor=self._executor,
        )

        plugin_ctx = self._collect_plugin_context()
        if self._on_pre_context:
            extra = await self._executor.run_hook(
                self._on_pre_context, msg.content, sid, msg.channel, msg.chat_id
            )
            if extra:
                plugin_ctx.append(str(extra))

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Use _bus_progress only when no on_progress was explicitly provided.
        # An explicit None means "run silently" (e.g. cron jobs via process_direct).
        effective_progress: Callable[..., Awaitable[None]] | None = (
            _bus_progress
            if on_progress is _UNSET
            else cast(Callable[..., Awaitable[None]] | None, on_progress)
        )

        final_content, new_msgs = await self.process_turn(
            sid,
            msg.content,
            channel=msg.channel,
            chat_id=msg.chat_id,
            media=msg.media if msg.media else None,
            plugin_context=plugin_ctx or None,
            on_progress=effective_progress,
            model=model or msg.model_override,
            publish_response=publish_response,
            **kwargs,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        if self._on_post_turn and new_msgs:
            asyncio.ensure_future(self._on_post_turn(new_msgs, sid, msg.channel, msg.chat_id))

        if publish_response and getattr(self._executor, "handles_response_send", False):
            # Executor took ownership of the send — nothing to return.
            return None

        if any(getattr(t, "sent_in_turn", False) for t in self.tools._tools.values()):
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        self._log.info("response_send", preview=preview)
        meta = dict(msg.metadata or {})
        meta.setdefault("session_key", sid)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        model: str | None = None,
        **kwargs: list[str] | None,
    ) -> str:
        """Process a message directly (for CLI or cron usage).

        ``model`` overrides the loop's default model for this turn only; pass
        ``None`` (the default) to inherit from the loop.

        Extra keyword arguments are forwarded to conversation.build_prompt,
        allowing callers to pass domain-specific context (e.g. skill_names,
        turn_context) without the loop needing to know about them.
        """
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        # Callers of ``process_direct`` read the returned content — they
        # don't want the executor to publish on their behalf.
        response = await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            model=model,
            publish_response=False,
            **kwargs,
        )
        return response.content if response else ""
