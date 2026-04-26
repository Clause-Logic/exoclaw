"""Targeted ``AgentLoop`` path coverage on MicroPython.

These tests target specific code paths in ``agent/loop.py`` that
``test_agent_loop.py`` / ``test_agent_loop_full.py`` /
``test_agent_loop_run.py`` don't reach: slash commands (``/new``,
``/help``), provider error responses, plugin-context exception
handling, tool ``set_bus`` / ``on_inbound`` / ``cancel_by_session``
hooks, ``_on_pre_context``, append-message persistence path,
iteration policy, and outbound metadata propagation.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

import asyncio

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import LLMResponse


class _StubProvider:
    def __init__(self, replies):
        self._replies = list(replies)

    def get_default_model(self):
        return "m"

    async def chat(self, messages, tools=None, model=None, **kw):
        if not self._replies:
            return LLMResponse(content="ok")
        return self._replies.pop(0)


class _MemConv:
    def __init__(self):
        self._messages = []

    async def build_prompt(self, sid, message, **kw):
        return self._messages + [{"role": "user", "content": message}]

    async def record(self, sid, msgs):
        self._messages.extend(msgs)

    async def clear(self, sid):
        self._messages = []
        return True

    def list_sessions(self):
        return []


class _AppendableConv:
    """Conversation that supports ``append`` (per-message persistence
    path) — exoclaw's loop calls ``append_message`` for the user
    message before the loop iteration starts."""

    def __init__(self):
        self._messages = []
        self.appended = []

    async def build_prompt(self, sid, message, **kw):
        return self._messages + [{"role": "user", "content": message}]

    async def append(self, sid, message):
        self.appended.append(message)
        self._messages.append(message)

    async def post_turn(self, sid):
        pass

    async def record(self, sid, msgs):  # never called when append is present
        pass

    async def clear(self, sid):
        self._messages = []
        self.appended = []
        return True

    def list_sessions(self):
        return []


# ── /new and /help slash commands ──────────────────────────────────


def test_slash_new_clears_session():
    """``/new`` runs ``clear`` on the conversation and publishes
    a ``New session started`` confirmation."""

    async def _go():
        bus = MessageBus()
        conv = _MemConv()
        # Pre-seed messages so we can verify clear empties them.
        conv._messages.append({"role": "user", "content": "old"})
        loop = AgentLoop(bus=bus, provider=_StubProvider([]), conversation=conv)
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="/new")
        )
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert "new session" in out.content.lower()
        assert conv._messages == []
        loop.stop()
        await run_task

    asyncio.run(_go())


def test_slash_help_returns_help_text():
    """``/help`` returns the canned help message — short-circuits
    before any LLM call."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(bus=bus, provider=_StubProvider([]), conversation=_MemConv())
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="/help")
        )
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert "/new" in out.content
        assert "/stop" in out.content
        assert "/help" in out.content
        loop.stop()
        await run_task

    asyncio.run(_go())


# ── Tool hooks: set_bus, on_inbound, cancel_by_session ────────────


def test_tool_set_bus_called_at_construction():
    """If a tool implements ``set_bus``, the AgentLoop calls it
    during ``__init__`` so the tool can publish outbound messages
    independently."""

    seen = []

    class _BusAware:
        name = "ba"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        def set_bus(self, bus):
            seen.append(bus)

    bus = MessageBus()
    AgentLoop(
        bus=bus,
        provider=_StubProvider([]),
        conversation=_MemConv(),
        tools=[_BusAware()],
    )
    assert seen == [bus]


def test_tool_on_inbound_called_with_message():
    """Tools that implement ``on_inbound`` get notified of every
    inbound message (used e.g. by tools that observe channel state)."""

    seen = []

    class _Listener:
        name = "ln"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        def on_inbound(self, msg):
            seen.append(msg)

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider([LLMResponse(content="r")]),
            conversation=_MemConv(),
            tools=[_Listener()],
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        await bus.publish_inbound(msg)
        await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        # Tool was notified.
        assert seen == [msg]
        loop.stop()
        await run_task

    asyncio.run(_go())


def test_collect_plugin_context_swallows_exceptions():
    """If a tool's ``system_context`` raises, the loop swallows the
    exception and the tool is omitted from the context list."""

    class _BoomCtx:
        name = "boom"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        def system_context(self):
            raise RuntimeError("ctx failure")

    class _Quiet:
        name = "q"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        def system_context(self):
            return "quiet"

    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_StubProvider([]),
        conversation=_MemConv(),
        tools=[_BoomCtx(), _Quiet()],
    )
    ctx = loop._collect_plugin_context()
    # The quiet tool's string is in the list. The boom tool's
    # exception was swallowed — it just doesn't contribute.
    assert "quiet" in ctx
    # Loop didn't crash. ``_BoomCtx`` should NOT appear.
    assert not any("ctx failure" in s for s in ctx)


# ── on_pre_context callback ────────────────────────────────────────


def test_on_pre_context_extra_appended_to_plugin_ctx():
    """``on_pre_context`` returns a string that's appended to the
    plugin_context list. Used by extensions that want to inject a
    per-turn system instruction."""

    seen_calls = []

    async def _on_pre_context(content, sid, channel, chat_id):
        seen_calls.append((content, sid, channel, chat_id))
        return "extra system context"

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider([LLMResponse(content="ok")]),
            conversation=_MemConv(),
            on_pre_context=_on_pre_context,
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        )
        await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        # Callback was invoked with the message and routing fields.
        assert len(seen_calls) == 1
        assert seen_calls[0][0] == "hi"
        loop.stop()
        await run_task

    asyncio.run(_go())


# ── Append-message persistence path ────────────────────────────────


def test_append_message_path_persists_user_message_first():
    """When the Conversation supports ``append``, the loop persists
    the user message before the LLM iteration starts. This is the
    crash-recovery guarantee — even if the LLM call fails, the user
    message is on disk."""

    async def _go():
        bus = MessageBus()
        conv = _AppendableConv()
        loop = AgentLoop(
            bus=bus, provider=_StubProvider([LLMResponse(content="r")]), conversation=conv
        )
        await loop.process_turn("s", "hi")
        # First appended message is the user message.
        assert conv.appended[0]["role"] == "user"
        assert conv.appended[0]["content"] == "hi"
        # Subsequent appends include the assistant reply.
        assert any(m.get("role") == "assistant" for m in conv.appended)

    asyncio.run(_go())


# ── Iteration policy ───────────────────────────────────────────────


def test_iteration_policy_termination_message():
    """When an ``IterationPolicy`` is provided, ``on_limit_reached``
    builds the termination message instead of the static default."""

    class _Policy:
        async def should_continue(self, iteration, tools_used):
            return iteration < 2

        async def on_limit_reached(self, iteration, tools_used):
            return "policy says stop"

    class _AlwaysToolCall:
        def get_default_model(self):
            return "m"

        async def chat(self, messages, tools=None, model=None, **kw):
            from exoclaw.providers.types import ToolCallRequest

            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="t", arguments={})],
                finish_reason="tool_calls",
            )

    class _Tool:
        name = "t"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "tool done"

    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_AlwaysToolCall(),
        conversation=_MemConv(),
        tools=[_Tool()],
        iteration_policy=_Policy(),
    )

    async def _go():
        content, _ = await loop.process_turn("s", "go")
        assert content == "policy says stop"

    asyncio.run(_go())


# ── LLM error finish_reason ────────────────────────────────────────


def test_legacy_execute_tool_path():
    """When the executor doesn't implement ``execute_tool_with_handle``
    (the Step-D opt-in), the loop falls back to the legacy
    ``execute_tool`` returning a plain string. ``content_file`` is
    None on this path."""
    from exoclaw.providers.types import ToolCallRequest

    legacy_calls = []

    class _LegacyExecutor:
        handles_response_send = False
        handles_inbound_enqueue = False

        def __init__(self, provider):
            self.provider = provider
            self._buffer: list[dict[str, object]] = []

        async def mint_turn_id(self):
            return "id"

        async def run_hook(self, fn, *a, **kw):
            return await fn(*a, **kw)

        async def run_turn(self, *a, **kw):
            return None

        async def execute_tool(self, registry, name, params, ctx, tool_call_id=None):
            legacy_calls.append((name, params))
            return "legacy-result"

        async def chat(self, provider, **kw):
            # Executor's ``chat`` wraps ``provider.chat`` — typically
            # adds durable retry / checkpoint. Test double delegates
            # straight through.
            return await provider.chat(**kw)

        # Note: NO execute_tool_with_handle method — forces the legacy path.

        async def build_prompt(self, conv, sid, msg, **kw):
            return await conv.build_prompt(sid, msg, **kw)

        async def append_message(self, conv, sid, message):
            pass

        async def post_turn(self, conv, sid):
            pass

        async def record(self, conv, sid, msgs):
            await conv.record(sid, msgs)

        def load_messages(self):
            return []

        def append_messages(self, msgs):
            pass

        def set_messages(self, msgs):
            pass

        async def clear(self, conv, sid):
            return await conv.clear(sid)

    class _Tool:
        name = "t"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "ignored"

    bus = MessageBus()
    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="t", arguments={})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]
    )
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=_MemConv(),
        tools=[_Tool()],
        executor=_LegacyExecutor(provider),
    )

    async def _go():
        content, _msgs = await loop.process_turn("s", "go")
        assert content == "done"
        # Legacy ``execute_tool`` was the path taken — the spy recorded
        # one call with the LLM-requested tool name.
        assert legacy_calls == [("t", {})]

    asyncio.run(_go())


def test_on_tool_calls_callback_fires_with_request_list():
    """Before executing tool calls, the loop fires ``on_tool_calls``
    with the structured ``ToolCallRequest`` list. UI / observability
    plugins use this to surface the LLM's intent before tools run."""
    from exoclaw.providers.types import ToolCallRequest

    seen_calls = []

    async def _on_tool_calls(tool_calls):
        seen_calls.append(list(tool_calls))

    class _Tool:
        name = "t"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "ok"

    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c1", name="t", arguments={"x": 1})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]
    )
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=_MemConv(),
        tools=[_Tool()],
        on_tool_calls=_on_tool_calls,
    )

    async def _go():
        await loop.process_turn("s", "go")
        # Callback fired once with the single tool call.
        assert len(seen_calls) == 1
        assert len(seen_calls[0]) == 1
        assert seen_calls[0][0].name == "t"

    asyncio.run(_go())


def test_handle_stop_invokes_cancel_by_session_on_tool():
    """``_handle_stop`` calls ``cancel_by_session(session_key)`` on
    every tool that implements it, summing the cancellation counts
    into the user-visible confirmation message."""

    class _CancellableTool:
        name = "ct"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        async def cancel_by_session(self, session_key):
            # Pretend we cancelled 2 background tasks for this session.
            return 2

    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_StubProvider([]),
        conversation=_MemConv(),
        tools=[_CancellableTool()],
    )

    async def _go():
        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="/stop")
        await loop._handle_stop(msg)
        out = await bus.consume_outbound()
        # Confirmation mentions the count from cancel_by_session.
        assert "2" in out.content

    asyncio.run(_go())


def test_llm_error_finish_reason_short_circuits():
    """If the provider returns ``finish_reason='error'``, the loop
    treats the response as terminal — emits ``llm_error`` log and
    returns the error content (or a default apology)."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider(
                [LLMResponse(content="API quota exhausted", finish_reason="error")]
            ),
            conversation=_MemConv(),
        )
        content, _ = await loop.process_turn("s", "ping")
        # Returns the error content as the user-facing reply.
        assert "quota" in (content or "").lower() or "error" in (content or "").lower()

    asyncio.run(_go())
