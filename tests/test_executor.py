"""Tests for the Executor protocol and DirectExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.executor import DirectExecutor, Executor


class TestDirectExecutorChat:
    async def test_delegates_to_provider(self) -> None:
        executor = DirectExecutor()
        provider = MagicMock()
        expected = MagicMock()
        provider.chat = AsyncMock(return_value=expected)

        result = await executor.chat(
            provider,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            reasoning_effort=None,
        )

        assert result is expected
        provider.chat.assert_awaited_once_with(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            reasoning_effort=None,
        )


class TestDirectExecutorExecuteTool:
    async def test_delegates_to_registry(self) -> None:
        executor = DirectExecutor()
        registry = MagicMock()
        registry.execute = AsyncMock(return_value="tool result")
        ctx = ToolContext(session_key="s:1", channel="cli", chat_id="1")

        result = await executor.execute_tool(registry, "my_tool", {"x": 1}, ctx)

        assert result == "tool result"
        registry.execute.assert_awaited_once_with("my_tool", {"x": 1}, ctx)

    async def test_none_ctx(self) -> None:
        executor = DirectExecutor()
        registry = MagicMock()
        registry.execute = AsyncMock(return_value="ok")

        result = await executor.execute_tool(registry, "t", {}, None)

        assert result == "ok"
        registry.execute.assert_awaited_once_with("t", {}, None)


class TestDirectExecutorMessageManagement:
    def test_set_and_load(self) -> None:
        executor = DirectExecutor()
        msgs = [{"role": "user", "content": "hi"}]
        executor.set_messages(msgs)
        assert executor.load_messages() == msgs

    def test_append_and_load(self) -> None:
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "hi"}])
        executor.append_messages([{"role": "assistant", "content": "hello"}])
        loaded = executor.load_messages()
        assert len(loaded) == 2
        assert loaded[0]["role"] == "user"
        assert loaded[1]["role"] == "assistant"

    def test_load_returns_copy(self) -> None:
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "hi"}])
        loaded = executor.load_messages()
        loaded.append({"role": "assistant", "content": "extra"})
        assert len(executor.load_messages()) == 1

    def test_set_replaces(self) -> None:
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "first"}])
        executor.set_messages([{"role": "user", "content": "second"}])
        loaded = executor.load_messages()
        assert len(loaded) == 1
        assert loaded[0]["content"] == "second"

    def test_starts_empty(self) -> None:
        executor = DirectExecutor()
        assert executor.load_messages() == []

    def test_two_instances_isolate_messages(self) -> None:
        """Two executors in the same task must not share a buffer.

        The ContextVar is per-instance for exactly this reason — a
        module-level var would make a second executor reset the first's
        state and vice versa.
        """
        a = DirectExecutor()
        b = DirectExecutor()
        a.set_messages([{"role": "user", "content": "a"}])
        b.set_messages([{"role": "user", "content": "b"}])
        assert [m["content"] for m in a.load_messages()] == ["a"]
        assert [m["content"] for m in b.load_messages()] == ["b"]
        a.append_messages([{"role": "assistant", "content": "a2"}])
        assert [m["content"] for m in a.load_messages()] == ["a", "a2"]
        assert [m["content"] for m in b.load_messages()] == ["b"]

    async def test_concurrent_turns_isolate_messages(self) -> None:
        """The shared executor singleton must not leak messages across
        concurrent turns.

        Regression for the incident where a cron's ``Learn about
        Stephen`` turn fired mid-flight and its messages ended up
        pre-filling the context of a peer ``/agent/call`` turn, leading
        the peer's LLM to hallucinate it was running the cron's skill
        and cross-contaminating both session JSONLs.

        Each asyncio.Task gets its own ContextVar binding, so both
        turns must observe exactly the messages they themselves wrote.
        """
        import asyncio

        executor = DirectExecutor()
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def turn(label: str, out: dict[str, list[dict[str, object]]]) -> None:
            executor.set_messages([{"role": "user", "content": f"{label}:user"}])
            entered.set()
            await proceed.wait()
            executor.append_messages([{"role": "assistant", "content": f"{label}:asst"}])
            out[label] = executor.load_messages()

        results: dict[str, list[dict[str, object]]] = {}
        t1 = asyncio.create_task(turn("a", results))
        # Let t1 set its messages and suspend.
        await entered.wait()
        entered.clear()
        t2 = asyncio.create_task(turn("b", results))
        await entered.wait()
        # Both tasks have now set distinct message lists; release them.
        proceed.set()
        await asyncio.gather(t1, t2)

        assert [m["content"] for m in results["a"]] == ["a:user", "a:asst"]
        assert [m["content"] for m in results["b"]] == ["b:user", "b:asst"]


class TestDirectExecutorPriorDeltaSplit:
    """Phase 2a invariants: the per-turn buffer is now split into
    read-only prior + mutable delta. ``load_messages`` concatenates.
    Tests below pin the internal structure so phase 2b (disk-backed
    prior) has a known-stable surface to build against.
    """

    def test_set_seeds_prior_not_delta(self) -> None:
        """``set_messages`` populates the prior half only. Delta stays
        empty so ``append_messages`` can't double-count the seeded
        messages."""
        executor = DirectExecutor()
        executor.set_messages([{"role": "system", "content": "sys"}])

        assert executor._get_prior() == [{"role": "system", "content": "sys"}]
        assert executor._get_delta() == []

    def test_append_grows_delta_not_prior(self) -> None:
        """``append_messages`` goes to delta. Prior stays exactly as
        ``set_messages`` left it so compaction / persistence code can
        rely on prior being read-only mid-turn."""
        executor = DirectExecutor()
        executor.set_messages([{"role": "system", "content": "sys"}])
        executor.append_messages([{"role": "assistant", "content": "ok"}])

        assert executor._get_prior() == [{"role": "system", "content": "sys"}]
        assert executor._get_delta() == [{"role": "assistant", "content": "ok"}]

    def test_set_clears_delta(self) -> None:
        """Compaction path: when ``set_messages`` is called mid-turn
        (e.g. after ContextWindowExceededError), the delta from the
        pre-compaction iterations must be cleared so the replacement
        list doesn't get double-counted by the next
        ``append_messages``."""
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "u"}])
        executor.append_messages([{"role": "assistant", "content": "a1"}])
        executor.append_messages([{"role": "tool", "content": "t1"}])

        # Simulate a mid-turn compaction call.
        executor.set_messages([{"role": "user", "content": "compacted"}])

        assert executor._get_prior() == [{"role": "user", "content": "compacted"}]
        assert executor._get_delta() == []
        assert executor.load_messages() == [{"role": "user", "content": "compacted"}]

    def test_load_is_fresh_list_over_prior_and_delta(self) -> None:
        """``load_messages`` allocates a new list so consumers can
        mutate the return (e.g. compaction callbacks) without racing
        with ``append_messages`` on the same shared state."""
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "u"}])
        executor.append_messages([{"role": "assistant", "content": "a"}])

        a = executor.load_messages()
        b = executor.load_messages()
        # Same content, different identity.
        assert a == b
        assert a is not b

        # Mutating one return doesn't change prior, delta, or a peer
        # return.
        a.append({"role": "tool", "content": "t"})
        assert len(executor.load_messages()) == 2
        assert len(executor._get_prior()) == 1
        assert len(executor._get_delta()) == 1

    def test_append_does_not_copy_prior(self) -> None:
        """The whole point of the split — ``append_messages`` must
        not touch the prior-history list. A phase 2b disk-backed
        prior implementation would pay a file-read every time
        ``append_messages`` ran if this invariant regressed.
        """
        executor = DirectExecutor()
        executor.set_messages([{"role": "system", "content": "sys"}])
        prior_ref = executor._get_prior()

        executor.append_messages([{"role": "assistant", "content": "a1"}])
        executor.append_messages([{"role": "tool", "content": "t1"}])

        # The exact same prior list object is still there, unmodified.
        assert executor._get_prior() is prior_ref
        assert prior_ref == [{"role": "system", "content": "sys"}]

    def test_set_prior_source_invokes_source_on_each_load(self) -> None:
        """``set_prior_source`` stores a callable. Each
        ``load_messages`` invokes it — the source can return
        different values across calls.

        A disk-backed source reads the JSONL fresh on each call; a
        dynamic in-memory source (like the one below) lets tests
        pin that invariant without touching the filesystem.
        """
        executor = DirectExecutor()
        counter = {"n": 0}

        def source() -> list[dict[str, object]]:
            counter["n"] += 1
            return [{"role": "user", "content": f"call-{counter['n']}"}]

        executor.set_prior_source(source)
        loaded1 = executor.load_messages()
        loaded2 = executor.load_messages()

        assert counter["n"] == 2, "source must be invoked on each load"
        assert loaded1 == [{"role": "user", "content": "call-1"}]
        assert loaded2 == [{"role": "user", "content": "call-2"}]

    def test_set_prior_source_clears_delta(self) -> None:
        """Analogous to ``set_messages``: installing a new prior
        source drops any delta that grew on the prior one. Otherwise
        sequential turns on the same task leak their delta state."""
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "t1"}])
        executor.append_messages([{"role": "assistant", "content": "t1-asst"}])

        executor.set_prior_source(lambda: [{"role": "user", "content": "t2"}])

        assert executor._get_delta() == []
        assert executor.load_messages() == [{"role": "user", "content": "t2"}]

    def test_set_prior_source_does_not_hold_list(self) -> None:
        """The whole point of phase 2b: a disk-backed source lets
        prior not be heap-resident between iterations. If the source
        closure doesn't capture the list, the list can be GC'd after
        each ``load_messages`` call.

        Simulate with a source that reads from a mutable wrapper —
        the prior_var itself holds only the source callable, not
        the list.
        """
        import sys

        executor = DirectExecutor()
        storage = {"msgs": [{"role": "user", "content": "on-demand"}]}

        def source() -> list[dict[str, object]]:
            return list(storage["msgs"])

        executor.set_prior_source(source)

        # The ContextVar should hold the callable, not a list.
        stored = executor._prior_var.get()
        assert callable(stored)
        assert stored is source

        # Swap the underlying storage — next load_messages reflects it.
        storage["msgs"] = [{"role": "user", "content": "swapped"}]
        assert executor.load_messages() == [{"role": "user", "content": "swapped"}]

        # sys.getrefcount is a sanity check that the list isn't
        # squirrelled away somewhere in the executor. The source
        # closure doesn't capture it (it re-reads storage each call).
        the_list = storage["msgs"]
        # Expected refs: our local ``the_list``, the dict ``storage["msgs"]``,
        # and Python's getrefcount-arg-temporary. No executor ref.
        assert sys.getrefcount(the_list) <= 3

    def test_set_messages_uses_prior_source_under_the_hood(self) -> None:
        """Back-compat: ``set_messages`` should route through the
        same source machinery. Otherwise the two APIs would diverge
        in subtle ways (delta clearing, etc.)."""
        executor = DirectExecutor()
        executor.set_messages([{"role": "user", "content": "hi"}])

        # prior_var holds a callable, not the list itself.
        stored = executor._prior_var.get()
        assert callable(stored)

        # Invoking it returns the snapshotted list.
        assert stored() == [{"role": "user", "content": "hi"}]

    def test_set_messages_snapshot_isolated_from_caller_mutation(self) -> None:
        """The back-compat ``set_messages`` closure captures a
        snapshot, not the caller's list reference. If a caller
        mutates its list after ``set_messages`` returns, the executor
        must not see the mutation — otherwise the pre-refactor
        "fresh list per call" guarantee would regress.
        """
        executor = DirectExecutor()
        msgs = [{"role": "user", "content": "original"}]
        executor.set_messages(msgs)

        msgs.append({"role": "assistant", "content": "injected"})
        assert executor.load_messages() == [{"role": "user", "content": "original"}]

    def test_set_prior_source_growth_independent_of_source_return_size(self) -> None:
        """Pin the phase 2b RAM invariant with a real bytes-level
        measurement: the executor's heap growth from
        ``set_prior_source`` must be roughly constant regardless of
        how large the list that source ``would`` return is.

        A regression that captured the source's return value (e.g. by
        pre-materialising the list inside the executor instead of
        storing the callable) would show growth proportional to the
        hypothetical return size. Two orders of magnitude of history
        size with near-identical executor growth is the invariant
        phase 2b was shipped for.

        Uses ``tracemalloc`` rather than the behavioural identity
        checks above because this is the actual RAM question —
        "do we hold the bytes or just the callable." Wrapped in
        ``gc.collect()`` calls to settle the interpreter's internal
        allocations between snapshots so the delta reflects the
        executor's state, not Python runtime noise.
        """
        import gc
        import tracemalloc

        # Two history sizes — one small, one 100× larger. Both exist
        # in memory before we measure; we're asking whether the
        # executor duplicates them or just holds a reference.
        small_history = [{"role": "user", "content": "x" * 100} for _ in range(10)]
        big_history = [{"role": "user", "content": "x" * 100} for _ in range(1000)]

        def _measure_growth(history: list[dict[str, object]]) -> int:
            executor = DirectExecutor()
            gc.collect()
            tracemalloc.start()
            snap_before = tracemalloc.take_snapshot()
            # The source allocates fresh dicts *on invocation* —
            # proportional to ``history`` size. If
            # ``set_prior_source`` accidentally invokes the source
            # or materialises its return (e.g. by storing
            # ``source()`` instead of ``source``), the measured
            # growth here will scale with history size and trip the
            # assertion. A lambda returning ``history`` directly
            # would not: the list is already allocated outside the
            # timed block, and any accidental invocation would
            # return an existing ref without new allocations.
            executor.set_prior_source(lambda: [dict(m) for m in history])
            snap_after = tracemalloc.take_snapshot()
            tracemalloc.stop()
            stats = snap_after.compare_to(snap_before, "lineno")
            return sum(s.size_diff for s in stats)

        growth_small = _measure_growth(small_history)
        growth_big = _measure_growth(big_history)

        # The delta between small-source and big-source growth
        # measures whether the executor duplicates source bytes.
        # With a correct implementation it's on the order of tens
        # of bytes (noise between runs); a regression that copied
        # would show a delta near ``len(big) - len(small)`` bytes.
        # 10 KB ceiling covers tracemalloc/GC noise with headroom
        # while still catching any real duplication.
        delta = abs(growth_big - growth_small)
        history_size_delta = len(str(big_history)) - len(str(small_history))
        assert delta < 10_000, (
            f"set_prior_source growth depends on source return size: "
            f"small-source growth {growth_small}B, big-source growth "
            f"{growth_big}B, delta {delta}B (history size delta was "
            f"~{history_size_delta}B). A regression that materialises "
            f"the source's return inside the executor would show up "
            f"here as delta ≈ history_size_delta."
        )

    def test_sequential_turns_on_same_task_dont_leak_delta(self) -> None:
        """Sequential turns on the same executor and same asyncio
        task share the ContextVar binding. Without the delta clear
        inside ``set_messages``, turn 2's ``load_messages`` would
        see turn 1's assistant/tool messages leaked in.
        """
        executor = DirectExecutor()

        # Turn 1: seed, append, load, turn ends.
        executor.set_messages([{"role": "user", "content": "t1-user"}])
        executor.append_messages([{"role": "assistant", "content": "t1-asst"}])
        assert executor.load_messages() == [
            {"role": "user", "content": "t1-user"},
            {"role": "assistant", "content": "t1-asst"},
        ]

        # Turn 2: seed via a new set_messages. Must wipe turn 1's delta.
        executor.set_messages([{"role": "user", "content": "t2-user"}])
        assert executor.load_messages() == [{"role": "user", "content": "t2-user"}]


class TestDirectExecutorBuildPrompt:
    async def test_delegates_to_conversation(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        expected = [{"role": "system", "content": "hi"}]
        conversation.build_prompt = AsyncMock(return_value=expected)

        result = await executor.build_prompt(
            conversation,
            "session:1",
            "hello",
            channel="cli",
            chat_id="1",
        )

        assert result is expected
        conversation.build_prompt.assert_awaited_once_with(
            "session:1",
            "hello",
            channel="cli",
            chat_id="1",
            media=None,
            plugin_context=None,
        )

    async def test_build_prompt_seeds_messages(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        conversation.build_prompt = AsyncMock(return_value=msgs)

        await executor.build_prompt(conversation, "s:1", "hi", channel="cli", chat_id="1")

        assert executor.load_messages() == msgs


class TestDirectExecutorRecord:
    async def test_delegates_to_conversation(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        conversation.record = AsyncMock()
        msgs = [{"role": "assistant", "content": "done"}]

        await executor.record(conversation, "session:1", msgs)

        conversation.record.assert_awaited_once_with("session:1", msgs)


class TestDirectExecutorAppendMessage:
    """Per-message persistence path. When the Conversation implements
    ``append`` as a real coroutine, ``append_message`` forwards to it.
    When it doesn't (e.g. a test MagicMock where any attribute access
    returns a Mock), ``append_message`` must be a no-op — otherwise
    every existing test that mocks Conversation would accidentally hit
    the per-message path and break on ``await`` of a non-coroutine.
    """

    async def test_forwards_to_conversation_append(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        conversation.append = AsyncMock()
        msg = {"role": "assistant", "content": "hi"}

        await executor.append_message(conversation, "session:1", msg)

        conversation.append.assert_awaited_once_with("session:1", msg)

    async def test_noop_when_append_is_plain_mock(self) -> None:
        """A MagicMock's ``.append`` is itself a MagicMock, not a
        coroutine — the executor must skip it, not ``await`` a non-
        awaitable and crash. Locks in backwards compat for the many
        existing tests that mock Conversation without asyncifying
        every attribute.
        """
        executor = DirectExecutor()
        conversation = MagicMock()  # append is a bare MagicMock, NOT AsyncMock

        # Must not raise. The existing mock test suite depends on this.
        await executor.append_message(conversation, "session:1", {"role": "user", "content": "x"})

        conversation.append.assert_not_called()

    async def test_noop_when_append_missing(self) -> None:
        executor = DirectExecutor()

        class _MinimalConversation:
            """No ``append`` method — legacy Conversation implementations
            that only support end-of-turn ``record`` still need to work."""

        await executor.append_message(
            _MinimalConversation(), "session:1", {"role": "user", "content": "x"}
        )


class TestDirectExecutorPostTurn:
    async def test_forwards_to_conversation_post_turn(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        conversation.post_turn = AsyncMock()

        await executor.post_turn(conversation, "session:1")

        conversation.post_turn.assert_awaited_once_with("session:1")

    async def test_noop_when_post_turn_is_plain_mock(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()

        await executor.post_turn(conversation, "session:1")

        conversation.post_turn.assert_not_called()


class TestDirectExecutorClear:
    async def test_delegates_to_conversation(self) -> None:
        executor = DirectExecutor()
        conversation = MagicMock()
        conversation.clear = AsyncMock(return_value=True)

        result = await executor.clear(conversation, "session:1")

        assert result is True
        conversation.clear.assert_awaited_once_with("session:1")


class TestDirectExecutorRunHook:
    async def test_calls_hook_function(self) -> None:
        executor = DirectExecutor()
        hook = AsyncMock(return_value="extra context")

        result = await executor.run_hook(hook, "arg1", "arg2", kw="val")

        assert result == "extra context"
        hook.assert_awaited_once_with("arg1", "arg2", kw="val")

    async def test_returns_none_from_hook(self) -> None:
        executor = DirectExecutor()
        hook = AsyncMock(return_value=None)

        result = await executor.run_hook(hook)

        assert result is None


class TestExecutorProtocol:
    def test_direct_executor_satisfies_protocol(self) -> None:
        assert isinstance(DirectExecutor(), Executor)


class TestCustomExecutorInLoop:
    """Verify the loop routes I/O through a custom executor."""

    async def test_chat_routed_through_executor(self) -> None:
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test"
        conversation = MagicMock()
        conversation.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        conversation.record = AsyncMock()

        # Custom executor that records calls
        _messages: list[dict[str, object]] = []
        executor = MagicMock(spec=DirectExecutor)
        response = MagicMock()
        response.has_tool_calls = False
        response.content = "hello"
        response.finish_reason = "stop"
        response.reasoning_content = None
        response.thinking_blocks = None
        executor.chat = AsyncMock(return_value=response)
        executor.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        executor.record = AsyncMock()
        executor.run_turn = AsyncMock(return_value=None)
        executor.set_messages = MagicMock(
            side_effect=lambda m: _messages.clear() or _messages.extend(m)
        )
        executor.load_messages = MagicMock(side_effect=lambda: list(_messages))
        executor.append_messages = MagicMock(side_effect=lambda m: _messages.extend(m))

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            executor=executor,
        )

        from exoclaw.bus.events import InboundMessage

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
        result = await loop._process_message(msg)

        assert result is not None
        assert result.content == "hello"
        # Verify executor.chat was called, not provider.chat directly
        executor.chat.assert_awaited_once()
        provider.chat.assert_not_called()

    async def test_clear_routed_through_executor(self) -> None:
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.bus.events import InboundMessage
        from exoclaw.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test"
        conversation = MagicMock()

        executor = MagicMock(spec=DirectExecutor)
        executor.clear = AsyncMock(return_value=True)

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            executor=executor,
        )

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="/new")
        result = await loop._process_message(msg)

        assert "New session" in result.content
        executor.clear.assert_awaited_once()
        # conversation.clear should NOT be called directly
        conversation.clear.assert_not_called()

    async def test_execute_tool_routed_through_executor(self) -> None:
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.bus.events import InboundMessage
        from exoclaw.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test"
        conversation = MagicMock()

        # Build an executor that returns tool calls on first chat, then final on second
        _messages: list[dict[str, object]] = []
        executor = MagicMock(spec=DirectExecutor)

        tc = MagicMock()
        tc.id = "call1"
        tc.name = "my_tool"
        tc.arguments = {"q": "test"}

        tool_resp = MagicMock()
        tool_resp.has_tool_calls = True
        tool_resp.content = None
        tool_resp.tool_calls = [tc]
        tool_resp.reasoning_content = None
        tool_resp.thinking_blocks = None

        final_resp = MagicMock()
        final_resp.has_tool_calls = False
        final_resp.content = "done"
        final_resp.finish_reason = "stop"
        final_resp.reasoning_content = None
        final_resp.thinking_blocks = None

        executor.chat = AsyncMock(side_effect=[tool_resp, final_resp])
        executor.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        executor.record = AsyncMock()
        executor.execute_tool = AsyncMock(return_value="tool output")
        executor.run_turn = AsyncMock(return_value=None)
        executor.set_messages = MagicMock(
            side_effect=lambda m: _messages.clear() or _messages.extend(m)
        )
        executor.load_messages = MagicMock(side_effect=lambda: list(_messages))
        executor.append_messages = MagicMock(side_effect=lambda m: _messages.extend(m))

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            conversation=conversation,
            executor=executor,
        )
        # Register a dummy tool so get_definitions returns something
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "test"
        tool.parameters = {}
        tool.sent_in_turn = False
        loop.tools.register(tool)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="do it")
        result = await loop._process_message(msg)

        assert result.content == "done"
        executor.execute_tool.assert_awaited_once()
