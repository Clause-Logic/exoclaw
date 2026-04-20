# exoclaw memory model: durability ≠ hibernation

This doc captures an architectural observation worth internalizing before
anyone expects exoclaw's durable executors to scale to "massive concurrent
subagents." The TL;DR: writing state to disk does not evacuate it from RAM.

## What lives on disk

Two durable stores cover exoclaw's execution state end-to-end:

- **Session JSONL** (`workspace/sessions/<session_key>.jsonl`) — written by
  `Conversation.record()` at the end of each turn. One line per message
  (user / assistant / tool). The authoritative history for *past* turns.
- **Durable-executor journal** (e.g. DBOS `operation_outputs` + `workflow_status`,
  or Temporal event history) — written after every `@DBOS.step` / activity
  completes. Journals step outputs so a crashed workflow can be rebooted
  and replayed without re-running completed side effects.

Together these are enough to **reboot the workflow from scratch** after a
crash. They are *not* designed to let the live process shed its working
set between awaits.

## What stays in RAM during an active turn

Even with both stores in place, an in-flight turn holds meaningful RAM:

1. **The current Python coroutine frame.** `_process_turn_inline` is
   suspended mid-`await`. Its locals, closures, executor reference,
   registry reference, and active skill state are all live Python
   objects. asyncio does not serialize coroutine frames between awaits.
2. **The in-flight HTTP request to the LLM.** Prompt body (often
   100–500 KB), TLS buffers, httpx connection state, response buffer.
   `@DBOS.step` wraps `provider.chat()`; the step body *runs* before it
   journals. While httpx is awaiting bytes, nothing is on disk.
3. **The latest tool result.** Returned synchronously, held by Python
   until the next `_chat_step` consumes it (and beyond, because of
   point 4). Large results — DB queries returning hundreds of rows, a
   batched spawn producing dozens of handles — sit in the message
   buffer at full size.
4. **The executor's `_messages` buffer.** This is the largest avoidable
   cost and deserves its own section.
5. **Loaded skills and hook-script content.** `SkillsLoader` caches
   `SKILL.md` content; hook scripts read from disk lazily but are
   pinned while active.

## The `_messages` buffer (the load-bearing duplication)

> **Note (concurrency):** before the ContextVar refactor, `_messages`
> was a plain instance attribute on the executor singleton. Two
> concurrent turns (e.g. a cron firing mid-IPC-turn) trampled each
> other's buffer, cross-contaminating LLM context and cross-writing
> session JSONLs. The current implementation binds the buffer to a
> `ContextVar` so each asyncio task gets its own list; this section
> below still applies to the RAM side of the story.


Every iteration of `AgentLoop._run_agent_loop` does:

```python
messages = self._executor.load_messages()          # copy of _messages
response = await self._executor.chat(
    provider, messages=messages, ...
)                                                  # json.dumps(messages)
                                                   # → HTTP request body
```

At the peak of each iteration there are **three concurrent copies** of
the full turn-local history:

- `self._executor._messages` — the accumulating authoritative buffer
- `messages = self._executor.load_messages()` — an explicit `list(...)`
  copy for this iteration
- the JSON-serialized request body inside httpx

And `_messages` *grows* — `append_messages([msg])` fires after every
assistant message and every tool result. A turn with 10 iterations and
a 50 KB tool result in the middle runs the buffer up to 500 KB–2 MB
*before* you account for the doubling/tripling at the httpx call.

This buffer is *not* a useless mirror of the JSONL — it holds the
*in-progress* messages of the current turn, which are only flushed to
disk when `Conversation.record()` runs at end of turn. Until then it
is the only place those messages live.

## Why durability doesn't save RAM today

- DBOS journals **step outputs**, not the Python coroutine. Between
  `await` points the task frame stays resident; DBOS has no mechanism to
  page it out and rehydrate on the next step completion.
- The durability contract is "if the process dies, a new process can
  replay the workflow from the journal." It is not "the live process
  can release the workflow's working set while it waits for IO."
- The JSONL is updated at turn boundaries, not per-message. Even if
  every message *were* flushed immediately, the executor still needs a
  message list in RAM to build the next prompt (every LLM call requires
  the full conversation in its request body).

## What would actually save RAM

Ordered by increasing effort:

1. **Flush every new message to the JSONL as it's produced** (after each
   `_chat_step`, after each `_tool_step`), not batched at `record()`.
2. **Convert the executor buffer from "full list" to "deltas + offset."**
   Executor tracks only messages produced this turn; prompt building
   reads prior history from disk (or a memory-mapped view) on demand.
3. **Stream the httpx request body instead of materializing the JSON.**
   Generator → chunked request. Avoids the third copy of the messages
   list during the send.
4. **Workflow-frame eviction** — Temporal's model. Between activities,
   the workflow worker drops the coroutine frame and reloads it from
   history when the next activity completes. Per-idle-workflow RAM
   approaches zero. DBOS could implement this but doesn't today.
5. **Per-step ephemeral processes** — each step runs in a fresh process,
   reads inputs from the journal, writes outputs, exits. Most aggressive;
   highest latency per step.

Combining (1)+(2)+(3) would get per-subagent RAM from a few MB of working
set down to a few hundred KB plus the unavoidable HTTP buffers. (4) and
(5) are the jumps to truly disk-backed execution, where concurrent-
subagent count is bounded by external resources (budget, task queue
depth) rather than the cgroup's memory limit.

## Practical implications

- "We have durability, so memory shouldn't matter" — false. Durability
  is a recovery property. RAM still matters for concurrent fan-out.
- `DBOS` + `Temporal` differ here: Temporal sheds workflow state between
  activities via its sandbox model; DBOS does not. If future durable
  executors want to compete with Temporal on fan-out scale, workflow
  eviction is the architectural jump, not "more durable storage."
- In-process concurrency is bounded by `total_cgroup_memory /
  per_subagent_working_set`. For the current implementation that's
  roughly 2–5 MB per active subagent, so a 512 MB cgroup caps at the
  low hundreds — not millions. Capping subagent concurrency in the
  spawner (see `SubagentSpawner.max_concurrent`) is the short-term
  answer; reducing per-subagent working set is the long-term one.

## When to revisit

Revisit if any of these become true:

- Subagent fan-out needs to scale past ~100 concurrent on a single host.
- Turn latency becomes dominated by httpx buffer allocation rather than
  the LLM call itself.
- An executor ships workflow-frame eviction (DBOS adding it, or migrating
  fan-out workloads to a Temporal executor).
- A tool legitimately needs to return multi-MB results (video transcripts,
  large document processing) and holding them in the message buffer
  starts starving other work.

Until then, the pattern is: keep concurrency caps at each layer (spawner,
provider, bus), accept the current per-subagent footprint, and treat
"more RAM" as a valid answer when capping isn't.
