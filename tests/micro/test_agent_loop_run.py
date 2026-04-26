"""``AgentLoop.run`` bus loop coverage on MicroPython.

Drives messages through the bus → ``run`` → ``_dispatch`` →
``process_turn`` → outbound publish chain. Tests that lifecycle
plus error / cancellation / context-overflow paths.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

import asyncio

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.types import (
    ContextWindowExceededError,
    LLMResponse,
    ToolCallRequest,
)


class _ProviderFromList:
    def __init__(self, replies):
        self._replies = list(replies)

    def get_default_model(self):
        return "m"

    async def chat(self, messages, tools=None, model=None, **kw):
        if not self._replies:
            return LLMResponse(content="default")
        return self._replies.pop(0)


class _RaisingProvider:
    """Raises an exception on the first call (and only call)."""

    def __init__(self, exc):
        self._exc = exc

    def get_default_model(self):
        return "m"

    async def chat(self, messages, tools=None, model=None, **kw):
        raise self._exc


class _MemConv:
    def __init__(self):
        self._messages = []

    async def build_prompt(self, session_id, message, **kw):
        return self._messages + [{"role": "user", "content": message}]

    async def record(self, session_id, msgs):
        self._messages.extend(msgs)

    async def clear(self, session_id):
        self._messages = []
        return True

    def list_sessions(self):
        return []


# ── run loop drives messages from bus to outbound ─────────────────


def test_run_loop_processes_message_and_publishes_outbound():
    """Driving the run loop end-to-end: publish an inbound message,
    expect ``run`` to consume it, dispatch through process_turn, and
    publish an outbound response."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_ProviderFromList([LLMResponse(content="reply")]),
            conversation=_MemConv(),
        )
        # Spawn the run loop in the background so we can publish into
        # the bus and observe the outbound side.
        run_task = asyncio.create_task(loop.run())
        # Small yield so the loop's first ``consume_inbound`` arms.
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        )
        # Read the outbound. The dispatch task gets scheduled inside
        # the run loop's iteration; allow a few yields for it to fire.
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert msg.content == "reply"
        # Cleanly stop the loop.
        loop.stop()
        await run_task

    asyncio.run(_go())


def test_stop_signal_cancels_run_loop():
    """``loop.stop()`` flips ``_running`` to False; the loop's next
    ``consume_inbound`` timeout drops it out of the while."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(bus=bus, provider=_ProviderFromList([]), conversation=_MemConv())
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        loop.stop()
        # Loop has 1.0s ``consume_inbound`` timeout — give it 2x.
        await asyncio.wait_for(run_task, timeout=3.0)

    asyncio.run(_go())


def test_run_loop_publishes_error_on_dispatch_exception():
    """If ``_dispatch`` raises (e.g. provider fails), the loop logs
    and publishes a generic apology outbound — doesn't crash the
    run loop."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_RaisingProvider(RuntimeError("provider down")),
            conversation=_MemConv(),
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        )
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert "error" in msg.content.lower()
        loop.stop()
        await run_task

    asyncio.run(_go())


def test_stop_command_via_bus():
    """A ``/stop`` inbound message routes to ``_handle_stop`` which
    publishes a confirmation outbound. No tasks were active so the
    confirmation reads ``No active task to stop``."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(bus=bus, provider=_ProviderFromList([]), conversation=_MemConv())
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="u", chat_id="c", content="/stop")
        )
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert "stopped" in msg.content.lower() or "no active task" in msg.content.lower()
        loop.stop()
        await run_task

    asyncio.run(_go())


# ── ContextWindowExceededError handler ────────────────────────────


def test_context_overflow_callback_compacts_and_continues():
    """When the provider raises ``ContextWindowExceededError`` and
    ``on_context_overflow`` is set, the callback's compacted list
    replaces the messages and the loop continues."""

    seen = []
    call_count = [0]

    class _OverflowOnceProvider:
        def get_default_model(self):
            return "m"

        async def chat(self, messages, tools=None, model=None, **kw):
            call_count[0] += 1
            seen.append(len(messages))
            if call_count[0] == 1:
                raise ContextWindowExceededError("too big")
            return LLMResponse(content="recovered")

    async def _compact(messages):
        # Drop everything but the last message.
        return messages[-1:]

    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_OverflowOnceProvider(),
        conversation=_MemConv(),
        on_context_overflow=_compact,
    )

    async def _go():
        content, _ = await loop.process_turn("s", "long")
        assert content == "recovered"
        # Two LLM calls — first overflowed, second succeeded with
        # the compacted message list.
        assert call_count[0] == 2

    asyncio.run(_go())


def test_context_overflow_no_callback_returns_apology():
    """Without ``on_context_overflow``, the loop emits a static
    apology message and exits the iteration — doesn't crash."""

    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_RaisingProvider(ContextWindowExceededError("too big")),
        conversation=_MemConv(),
    )

    async def _go():
        content, _ = await loop.process_turn("s", "long")
        assert content is not None
        # The static apology mentions context window in some form.
        assert "context" in content.lower() or "window" in content.lower()

    asyncio.run(_go())


# ── _fmt helper for tool call logging ─────────────────────────────


def test_fmt_tool_call_truncates_long_args():
    """The internal ``_fmt`` helper inside ``_run_agent_loop`` builds
    a single-line representation of a ToolCallRequest for log lines.
    Exercise it indirectly by triggering a turn with a tool call and
    making sure logging doesn't blow up."""

    async def _go():
        class _Stamp:
            name = "stamp"
            description = "x"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, **kw):
                return "ok"

        provider = _ProviderFromList(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="c1",
                            name="stamp",
                            arguments={"k": "v" * 1000},  # Long args — exercises trunc.
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="done"),
            ]
        )
        bus = MessageBus()
        loop = AgentLoop(bus=bus, provider=provider, conversation=_MemConv(), tools=[_Stamp()])
        content, _ = await loop.process_turn("s", "go")
        assert content == "done"

    asyncio.run(_go())


# ── _dispatch with system message vs user message ─────────────────


def test_durable_inbound_enqueue_wires_set_inbound_hook():
    """When the executor advertises ``handles_inbound_enqueue=True``,
    the loop wires the bus's ``set_inbound_hook`` to the executor's
    ``enqueue_inbound``. ``publish_inbound`` then forwards directly
    to the executor's durable journal."""

    enqueue_calls = []

    class _DurableExecutor:
        handles_inbound_enqueue = True
        handles_response_send = False

        async def enqueue_inbound(self, msg):
            enqueue_calls.append(msg)

        # Stubs for the rest of the Executor protocol — never invoked
        # in this test; the loop only constructs the executor.
        async def mint_turn_id(self):
            return "id"

        async def run_hook(self, fn, *a, **kw):
            return await fn(*a, **kw)

        async def run_turn(self, *a, **kw):
            return None

        async def execute_tool(self, *a, **kw):
            return ""

        async def build_prompt(self, conv, sid, msg, **kw):
            return [{"role": "user", "content": msg}]

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

    bus = MessageBus()
    AgentLoop(
        bus=bus,
        provider=_ProviderFromList([]),
        conversation=_MemConv(),
        executor=_DurableExecutor(),
    )

    async def _go():
        msg = InboundMessage(channel="x", sender_id="u", chat_id="c", content="hi")
        await bus.publish_inbound(msg)
        # Hook fired → message went to executor, NOT the queue.
        assert enqueue_calls == [msg]
        assert bus.inbound.empty()

    asyncio.run(_go())


def test_durable_inbound_enqueue_missing_wiring_raises():
    """If the executor sets ``handles_inbound_enqueue=True`` without
    actually exposing ``enqueue_inbound``, the AgentLoop constructor
    raises a ``TypeError`` rather than silently falling back — the
    fallback would re-open the durability gap the opt-in is meant
    to close."""

    class _BrokenExecutor:
        handles_inbound_enqueue = True
        handles_response_send = False
        # No enqueue_inbound method.

    bus = MessageBus()
    raised = []
    try:
        AgentLoop(
            bus=bus,
            provider=_ProviderFromList([]),
            conversation=_MemConv(),
            executor=_BrokenExecutor(),
        )
    except TypeError as e:
        raised.append(str(e))
    assert raised, "expected TypeError for missing enqueue_inbound wiring"
    assert "enqueue_inbound" in raised[0]


def test_strip_think_with_progress_callback():
    """``on_progress`` callback receives stripped reasoning when the
    response includes ``<think>...</think>`` tags before tool calls.
    Exercises the ``_strip_think`` call inside the tool-dispatch
    iteration (line 353)."""

    progress_calls = []

    async def _on_progress(content, *, tool_hint=False):
        progress_calls.append((content, tool_hint))

    class _Stamp:
        name = "stamp"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return "ok"

    provider = _ProviderFromList(
        [
            LLMResponse(
                content="<think>let me think</think>I'll use stamp",
                tool_calls=[ToolCallRequest(id="c", name="stamp", arguments={})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]
    )
    bus = MessageBus()
    loop = AgentLoop(bus=bus, provider=provider, conversation=_MemConv(), tools=[_Stamp()])

    async def _go():
        await loop.process_turn("s", "go", on_progress=_on_progress)
        # ``_strip_think`` removed the think block; the cleaned
        # text reached on_progress at least once before the tool ran.
        cleaned = [c for c, h in progress_calls if "think" not in c.lower()]
        assert cleaned, "expected some progress message without think tags, got {!r}".format(
            progress_calls
        )

    asyncio.run(_go())


def test_dispatch_system_message_branch():
    """Inbound messages with ``metadata['_system']=True`` take the
    system-message dispatch branch — different log line, same end
    result (publishes outbound)."""

    async def _go():
        bus = MessageBus()
        loop = AgentLoop(
            bus=bus,
            provider=_ProviderFromList([LLMResponse(content="sys reply")]),
            conversation=_MemConv(),
        )
        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(
                channel="cli",
                sender_id="system",
                chat_id="c",
                content="system event",
                metadata={"_system": True},
            )
        )
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert msg.content == "sys reply"
        loop.stop()
        await run_task

    asyncio.run(_go())
