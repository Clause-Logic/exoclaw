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

from exoclaw.agent.hooks import (
    BeforeFinishResult,
    BeforeToolResult,
)
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import LLMResponse, ToolCallRequest


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


def test_system_channel_routes_to_chat_id_subchannel():
    """An inbound with ``channel="system"`` is the path cron jobs /
    background workers use to inject a turn. The loop parses the
    real channel out of ``chat_id`` (``"cli:abc"`` → channel=cli,
    chat_id=abc) and emits an outbound back on that subchannel."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider([LLMResponse(content="bg result")]),
            conversation=_MemConv(),
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(
                channel="system",
                sender_id="cron",
                chat_id="cli:scheduled-job",
                content="run the thing",
            )
        )
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        # Outbound goes back on the parsed subchannel.
        assert out.channel == "cli"
        assert out.chat_id == "scheduled-job"
        assert out.content == "bg result"
        # ``session_key`` metadata is set by the system path.
        assert out.metadata.get("session_key") == "cli:scheduled-job"
        loop.stop()
        await run_task

    asyncio.run(_go())


def test_system_channel_default_subchannel_when_no_colon():
    """``chat_id`` without a colon falls back to ``cli:`` prefix —
    same default as the regular CLI path."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider([LLMResponse(content="ok")]),
            conversation=_MemConv(),
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(
                channel="system",
                sender_id="bg",
                chat_id="bare-id",
                content="do thing",
            )
        )
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert out.channel == "cli"
        assert out.chat_id == "bare-id"
        loop.stop()
        await run_task

    asyncio.run(_go())


def test_on_max_iterations_callback_fires():
    """When the agent loop hits ``max_iterations`` and a
    ``ToolContext`` is active, ``on_max_iterations`` is fired with
    the routing fields. Used by plugins that want to surface
    "I gave up" telemetry per session."""
    from exoclaw.providers.types import ToolCallRequest

    seen = []

    async def _on_max(session_key, channel, chat_id):
        seen.append((session_key, channel, chat_id))

    class _LoopForever:
        name = "lp"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "still going"

    class _AlwaysCall:
        def get_default_model(self):
            return "m"

        async def chat(self, messages, tools=None, model=None, **kw):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="lp", arguments={})],
                finish_reason="tool_calls",
            )

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_AlwaysCall(),
            conversation=_MemConv(),
            tools=[_LoopForever()],
            max_iterations=2,
            on_max_iterations=_on_max,
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="go")
        )
        await asyncio.wait_for(bus.consume_outbound(), timeout=3.0)
        # ``on_max_iterations`` may run as ``ensure_future`` —
        # yield enough times to let it settle.
        for _ in range(3):
            await asyncio.sleep(0.01)
        assert len(seen) == 1
        assert seen[0][1] == "cli"
        assert seen[0][2] == "c"
        loop.stop()
        await run_task

    asyncio.run(_go())


def test_streaming_tool_attaches_content_file_to_tool_message():
    """When a streaming tool returns a file-backed result, the
    agent loop attaches the path to the assistant-visible tool
    message under the ``_content_file`` transport key. Providers
    use this to stream from disk into the LLM request body.

    Mirrors the memory-model.md Step D contract end-to-end on MP."""
    from exoclaw.providers.types import ToolCallRequest

    class _Streaming:
        name = "stream"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

        async def execute_streaming(self, **kw):
            yield "x" * 200
            yield "y" * 200

    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="stream", arguments={})],
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
        tools=[_Streaming()],
    )

    async def _go():
        _content, msgs = await loop.process_turn("s", "go")
        # The tool message carries ``_content_file`` pointing at the scratch path.
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert tool_msgs
        # The transport key is set with the str path. (CPython uses ``str``
        # on a Path object; MP just propagates the string.)
        path = tool_msgs[0].get("_content_file")
        assert path is not None
        assert isinstance(path, str)
        # Cleanup the scratch file.
        import os as _os

        try:
            _os.remove(path)
        except OSError:
            pass

    asyncio.run(_go())


def test_on_tool_result_callback_fires_after_each_tool():
    """``on_tool_result`` is invoked with ``(tool_call, result)``
    after each tool finishes — used by streaming UIs to surface
    intermediate state to the user."""
    from exoclaw.providers.types import ToolCallRequest

    seen = []

    async def _on_tool_result(tool_call, result):
        seen.append((tool_call.name, result))

    class _Echo:
        name = "echo"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "echoed"

    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="echo", arguments={})],
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
        tools=[_Echo()],
        on_tool_result=_on_tool_result,
    )

    async def _go():
        await loop.process_turn("s", "go")
        assert seen == [("echo", "echoed")]

    asyncio.run(_go())


def test_reasoning_and_thinking_blocks_with_tool_calls():
    """When the LLM returns BOTH tool calls AND reasoning / thinking
    fields, the loop attaches them to the assistant message that
    accompanies the tool calls. Covers the
    ``msg["reasoning_content"] = ...`` and
    ``msg["thinking_blocks"] = ...`` branches in the tool-call
    iteration (different code path from the final-answer branch)."""
    from exoclaw.providers.types import ToolCallRequest

    class _Echo:
        name = "echo"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "ok"

    thinking_blocks = [{"type": "thinking", "text": "deciding to call echo"}]
    provider = _StubProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c", name="echo", arguments={})],
                finish_reason="tool_calls",
                reasoning_content="step 1: call echo",
                thinking_blocks=thinking_blocks,
            ),
            LLMResponse(content="done"),
        ]
    )
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=_MemConv(),
        tools=[_Echo()],
    )

    async def _go():
        _content, msgs = await loop.process_turn("s", "go")
        # Find the assistant message that carries the tool call.
        asst_with_tools = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
        assert asst_with_tools
        m = asst_with_tools[0]
        assert m.get("reasoning_content") == "step 1: call echo"
        assert m.get("thinking_blocks") == thinking_blocks

    asyncio.run(_go())


def test_thinking_blocks_attached_to_assistant_message():
    """A provider response with ``thinking_blocks`` (anthropic-style
    extended thinking) propagates to the persisted assistant message
    so the next turn's prompt can replay them. Covers the
    ``msg["thinking_blocks"] = response.thinking_blocks`` branch."""

    async def _go():
        thinking = [{"type": "thinking", "text": "step-by-step"}]
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider([LLMResponse(content="answer", thinking_blocks=thinking)]),
            conversation=_MemConv(),
        )
        _content, msgs = await loop.process_turn("s", "ask")
        asst = [m for m in msgs if m.get("role") == "assistant"]
        assert asst
        assert asst[0].get("thinking_blocks") == thinking

    asyncio.run(_go())


def test_executor_handles_response_send_returns_none():
    """If the executor advertises ``handles_response_send=True``, the
    loop's ``_process_message`` returns ``None`` and ``_dispatch``
    publishes an empty placeholder for cli channels (avoids a hung
    CLI prompt). Covers line 766's elif msg.channel == "cli" path."""

    class _SendingExecutor:
        handles_response_send = True
        handles_inbound_enqueue = False

        async def mint_turn_id(self):
            return "id"

        async def run_hook(self, fn, *a, **kw):
            return await fn(*a, **kw)

        async def run_turn(self, *a, **kw):
            return None

        async def execute_tool(self, *a, **kw):
            return ""

        async def execute_tool_with_handle(self, *a, **kw):
            from exoclaw.executor import ToolResult

            return ToolResult(content="", content_file=None)

        async def chat(self, provider, **kw):
            return await provider.chat(**kw)

        async def build_prompt(self, conv, sid, msg, **kw):
            return await conv.build_prompt(sid, msg, **kw)

        async def append_message(self, conv, sid, message):
            pass

        async def post_turn(self, conv, sid):
            pass

        async def record(self, conv, sid, msgs):
            pass

        def load_messages(self):
            return []

        def append_messages(self, msgs):
            pass

        def set_messages(self, msgs):
            pass

        async def clear(self, conv, sid):
            return True

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider([LLMResponse(content="reply")]),
            conversation=_MemConv(),
            executor=_SendingExecutor(),
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        )
        # Loop emits empty-content outbound when executor took over the send.
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        # CLI channel always gets an outbound, even if empty — avoids a hung prompt.
        assert out.channel == "cli"
        loop.stop()
        await run_task

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


def test_on_before_finish_injects_then_ends():
    """``on_before_finish`` returning a follow-up re-prompts the model in
    place (loop continues); returning None ends the turn. Covers the
    no-tool-calls before-finish branch in ``_run_agent_loop`` — both the
    inject-and-continue path and the satisfied/end path."""

    async def _go():
        seen = []

        async def on_before_finish(final, tools_used, session_key):
            seen.append(final)
            # Nudge once, then accept.
            return "keep going" if len(seen) == 1 else None

        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_StubProvider([LLMResponse(content="partial"), LLMResponse(content="done")]),
            conversation=_MemConv(),
            on_before_finish=on_before_finish,
        )
        content, _ = await loop.process_turn("s", "go")
        # First stop nudged → loop continued; ended on the second response.
        assert content == "done"
        assert seen == ["partial", "done"]

    asyncio.run(_go())


def test_on_steer_before_tools_skips_unstarted_calls():
    """A steering message received after tool selection closes every
    unstarted call, then re-prompts the current turn with that message."""

    seen = []

    async def on_steer(session_id):
        seen.append(session_id)
        return ["use the local file"] if len(seen) == 2 else []

    class _MustNotRun:
        name = "side_effect"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            raise AssertionError("steering should skip this tool")

    async def _go():
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider(
                [
                    LLMResponse(
                        content="calling tools",
                        tool_calls=[
                            ToolCallRequest(id="one", name="side_effect", arguments={}),
                            ToolCallRequest(id="two", name="side_effect", arguments={}),
                        ],
                    ),
                    LLMResponse(content="updated"),
                ]
            ),
            conversation=_MemConv(),
            tools=[_MustNotRun()],
            on_steer=on_steer,
        )
        content, messages = await loop.process_turn("steer:before", "research it")
        assert content == "updated"
        assert [message["role"] for message in messages[-5:-1]] == [
            "assistant",
            "tool",
            "tool",
            "user",
        ]
        assert "Skipped because" in messages[-4]["content"]
        assert "Skipped because" in messages[-3]["content"]
        assert messages[-2]["content"] == "use the local file"

    asyncio.run(_go())


def test_on_steer_after_tool_skips_only_remaining_calls():
    """Steering after a completed tool preserves its result but skips
    sibling calls that have not started yet."""

    seen = []

    async def on_steer(session_id):
        seen.append(session_id)
        return ["stop now"] if len(seen) == 3 else []

    class _Counter:
        name = "count"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        async def execute(self, **kw):
            self.calls += 1
            return "first result"

    async def _go():
        tool = _Counter()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider(
                [
                    LLMResponse(
                        content="calling tools",
                        tool_calls=[
                            ToolCallRequest(id="one", name="count", arguments={}),
                            ToolCallRequest(id="two", name="count", arguments={}),
                        ],
                    ),
                    LLMResponse(content="stopped"),
                ]
            ),
            conversation=_MemConv(),
            tools=[tool],
            on_steer=on_steer,
        )
        content, messages = await loop.process_turn("steer:after-tool", "do it")
        assert content == "stopped"
        assert tool.calls == 1
        assert [message["content"] for message in messages[-4:-1]] == [
            "first result",
            "Skipped because a new user message arrived before this tool started.",
            "stop now",
        ]

    asyncio.run(_go())


def test_on_steer_after_final_response_reprompts_turn():
    """A message arriving while a final response is produced starts another
    model iteration within the same turn."""

    seen = []

    async def on_steer(session_id):
        seen.append(session_id)
        return ["make it short"] if len(seen) == 2 else []

    async def _go():
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider([LLMResponse(content="long"), LLMResponse(content="short")]),
            conversation=_MemConv(),
            on_steer=on_steer,
        )
        content, messages = await loop.process_turn("steer:final", "answer")
        assert content == "short"
        assert [message["content"] for message in messages[-3:-1]] == ["long", "make it short"]

    asyncio.run(_go())


def test_on_steer_errors_and_invalid_values_do_not_break_turn():
    """The host-side steering source is best-effort: a bad drain leaves the
    current turn usable rather than surfacing an unrelated error."""

    seen = []

    async def on_steer(session_id):
        seen.append(session_id)
        if len(seen) == 1:
            raise RuntimeError("store unavailable")
        return ["text", 1]

    async def _go():
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider([LLMResponse(content="done")]),
            conversation=_MemConv(),
            on_steer=on_steer,
        )
        content, _messages = await loop.process_turn("steer:error", "continue")
        assert content == "done"

    asyncio.run(_go())


# ── lifecycle hooks (exoclaw.agent.hooks) under MicroPython ──────────────────


class _HookConv:
    """Conversation that delegates the decider seams + surfaces run_context."""

    def __init__(self, before_tool=None, before_finish=None, run_ctx=None):
        self._bt = before_tool
        self._bf = before_finish
        self._run_ctx = run_ctx or {}

    async def build_prompt(self, sid, message, **kw):
        return [{"role": "user", "content": message}]

    async def record(self, sid, msgs):
        pass

    async def clear(self, sid):
        return True

    def list_sessions(self):
        return []

    def run_context(self):
        return self._run_ctx

    async def before_tool(self, ctx):
        return await self._bt(ctx) if self._bt else None

    async def before_finish(self, ctx):
        return await self._bf(ctx) if self._bf else None


class _RecTool:
    name = "do"
    description = "d"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.received = None
        self.executed = False

    async def execute(self, **kwargs):
        self.executed = True
        self.received = kwargs
        return "ok"


def _tool_call_resp(name="do", args=None):
    return LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id="t1", name=name, arguments=args or {})],
        finish_reason="tool_calls",
    )


def test_before_tool_decider_stamps_from_run_context():
    """A before_tool decider reads run_context and stamps the authoritative
    value onto the tool args."""

    async def _go():
        tool = _RecTool()

        async def stamp(ctx):
            p = dict(ctx.params or {})
            p["cycle_id"] = ctx.run_context.get("cycle_id")
            return BeforeToolResult(params=p)

        conv = _HookConv(before_tool=stamp, run_ctx={"cycle_id": "C1"})
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider(
                [_tool_call_resp(args={"q": "x"}), LLMResponse(content="final")]
            ),
            conversation=conv,
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.received == {"q": "x", "cycle_id": "C1"}

    asyncio.run(_go())


def test_before_tool_decider_vetoes():
    async def _go():
        tool = _RecTool()

        async def veto(ctx):
            return BeforeToolResult(block=True, block_reason="no")

        conv = _HookConv(before_tool=veto)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider([_tool_call_resp(), LLMResponse(content="final")]),
            conversation=conv,
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed is False

    asyncio.run(_go())


def test_before_finish_decider_injects():
    async def _go():
        seen = []

        async def nudge(ctx):
            seen.append(1)
            return BeforeFinishResult(continue_message="keep going" if len(seen) == 1 else None)

        conv = _HookConv(before_finish=nudge)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider([LLMResponse(content="partial"), LLMResponse(content="done")]),
            conversation=conv,
        )
        out = await loop.process_direct("go")
        assert out == "done"
        assert len(seen) == 2

    asyncio.run(_go())


class _ThrowingHookConv:
    """Decider seams (and run_context) raise — the loop must treat them as
    no-ops, never crash the turn."""

    def __init__(self, raise_before_tool):
        self._raise_before_tool = raise_before_tool

    async def build_prompt(self, sid, message, **kw):
        return [{"role": "user", "content": message}]

    async def record(self, sid, msgs):
        pass

    async def clear(self, sid):
        return True

    def list_sessions(self):
        return []

    async def before_tool(self, ctx):
        if self._raise_before_tool:
            raise RuntimeError("boom")
        return None

    def run_context(self):
        raise RuntimeError("boom")


def test_throwing_before_tool_is_noop():
    async def _go():
        tool = _RecTool()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider(
                [_tool_call_resp(args={"q": "x"}), LLMResponse(content="final")]
            ),
            conversation=_ThrowingHookConv(raise_before_tool=True),
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed

    asyncio.run(_go())


def test_throwing_run_context_is_noop():
    async def _go():
        tool = _RecTool()
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_StubProvider(
                [_tool_call_resp(args={"q": "x"}), LLMResponse(content="final")]
            ),
            conversation=_ThrowingHookConv(raise_before_tool=False),
            tools=[tool],
        )
        out = await loop.process_direct("go")
        assert out == "final"
        assert tool.executed

    asyncio.run(_go())
