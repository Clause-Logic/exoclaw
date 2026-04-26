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
> `ContextVar` so each asyncio task gets its own list.
>
> **Note (post-phase-2a):** the single `_messages_var` described
> below was split into `_prior_var` (a `PriorSource` callable) +
> `_delta_var` (a list of messages produced this turn) in exoclaw
> 0.19.1. The structural story this section describes still holds
> — at the moment of the httpx send there are still three copies —
> but "the buffer" is now two ContextVars. See "Current state"
> below for what actually shipped.


Every iteration of `AgentLoop._run_agent_loop` does:

```python
messages = self._executor.load_messages()          # copy of the per-turn buffer
response = await self._executor.chat(
    provider, messages=messages, ...
)                                                  # json.dumps(messages)
                                                   # → HTTP request body
```

At the peak of each iteration there are **three concurrent copies** of
the full turn-local history:

- the executor's per-turn message buffer (ContextVar-backed) — the
  accumulating authoritative list
- `messages = self._executor.load_messages()` — an explicit `list(...)`
  copy for this iteration
- the JSON-serialized request body inside httpx

And the buffer *grows* — `append_messages([msg])` fires after every
assistant message and every tool result. A turn with 10 iterations and
a 50 KB tool result in the middle runs the buffer up to 500 KB–2 MB
*before* you account for the doubling/tripling at the httpx call.

This buffer is *not* a useless mirror of the JSONL — it holds the
*in-progress* messages of the current turn. Historically they were
only flushed to disk when `Conversation.record()` ran at end of
turn, so the buffer was the only place those messages lived. Phase
1 (see below) changed that: each message is flushed per-produce
via `Conversation.append`, so a mid-turn crash no longer loses the
transcript. The buffer still exists in-memory (the LLM needs the
full conversation as its next request body), it just isn't the
single source of truth for durability anymore.

## Why durability doesn't save RAM today

- DBOS journals **step outputs**, not the Python coroutine. Between
  `await` points the task frame stays resident; DBOS has no mechanism to
  page it out and rehydrate on the next step completion.
- The durability contract is "if the process dies, a new process can
  replay the workflow from the journal." It is not "the live process
  can release the workflow's working set while it waits for IO."
- Per-message flushing (phase 1, shipped) fixed the crash-recovery
  hole — messages make it to disk as they're produced, not at
  turn-end. But this didn't reduce RAM by itself: the executor
  still needs a message list in RAM to build the next prompt
  (every LLM call requires the full conversation in its request
  body). The RAM reduction depends on phase 2+3 — see "Current
  state" and "Next steps" below.

## Current state: what's shipped, what it bought

Three phases on a multi-phase plan toward near-zero working set. The
first two are in PyPI and deployed to openclaw; phase 3 is not started.

### Phase 1 — per-message JSONL append [SHIPPED]

*exoclaw 0.19.0 · exoclaw-conversation 0.15.0 · exoclaw-executor-dbos 0.12.0*

`Conversation` gains an opt-in `AppendableConversation` extension:
`append(session_id, message)` + `post_turn(session_id)`. The agent
loop detects the capability (via `asyncio.iscoroutinefunction`
inspection of `append`) and switches from end-of-turn
`record(new_messages)` to per-message flush: each assistant response,
tool result, and the incoming user message writes one line to the
session JSONL as it's produced. `post_turn` handles end-of-turn
hooks; `record` is skipped entirely on this path.

`DBOSExecutor.append_message` wraps each flush in a `@DBOS.step`, so
on workflow replay the step's journaled completion returns without
re-writing the same message to disk. Matches PR #44's at-least-once
posture for the final-reply send — duplicate write on crash between
publish and mark, never a silent drop.

**What it bought:** crash-recoverability. A mid-turn OOM no longer
loses the partially-produced transcript. The deployed executor on
openclaw stopped orphaning subagent completions after DBOS recovery
(see the 2026-04-23 feed-digest-retry incident post-mortem). Did
**not** reduce RAM — the in-memory buffer still held the full turn
state because subsequent iterations still needed it for prompt
construction. Phase 1 is the foundation that made phase 2 possible.

### Phase 2a — prior/delta split [SHIPPED]

*exoclaw 0.19.1 · exoclaw-executor-dbos 0.13.0*

`DirectExecutor` and `DBOSExecutor` split the single `_messages_var`
ContextVar into a `_prior_var` (read-only, set at turn start) and a
`_delta_var` (appended to mid-turn). `load_messages()` concatenates
prior + delta into a fresh list; `set_messages` seeds prior and
clears delta; `append_messages` extends delta only. Prior is
guaranteed not to be touched by the append path.

**What it bought:** nothing directly. Pure structural refactor.
Enabled phase 2b by establishing the read-only-prior invariant.

### Phase 2b — disk-backed `PriorSource` [SHIPPED]

*exoclaw 0.20.0 · exoclaw-conversation 0.16.0 · exoclaw-executor-dbos 0.13.1*

`_prior_var` now stores a callable `PriorSource` rather than a
`list[dict]`. `DirectExecutor.set_prior_source(source)` installs a
lazy source; `set_messages` is a snapshot-closure wrapper (back-compat).
`load_messages` invokes the source each call.

`DefaultConversation` grows `load_persisted_history(session_id)` (sync,
no consolidation side effects) — the hook a lazy source needs to
re-fetch the history slice from session state each iteration.

`DBOSExecutor.build_prompt` auto-detects the capability and
installs a disk-backed source by locating the history slice in
`build_prompt`'s return. Slice match is by **dict equality** (not
`id()`) — critical for the real `DefaultConversation`, whose
`session.get_history` strips timestamps and returns fresh dicts per
call. An earlier id-based version shipped in 0.13.0 always fell back
to snapshot mode in production; 0.13.1 fixes that.

`DirectExecutor.build_prompt` gained the same auto-wire in exoclaw
0.23.0 (the helper `_build_lazy_prior_source` lives in
`exoclaw.executor` and is shared between executors). Until 0.23.0
the pass-through executor would always snapshot via `set_messages`,
so the phase 2b RAM win was DBOS-only — non-DBOS deployments,
loadtest scripts, and tests using `DirectExecutor` had to call
`set_prior_source` manually to realise it.

**What it bought, in principle:** the history slice (typically the
bulk of prompt size) would not be held in `_prior_var` between LLM
iterations — re-read on demand per `load_messages` call. Combined
with phase 1 flushing, the between-iteration heap footprint would
drop to roughly (delta-so-far + httpx-connection-state).

**What it bought in practice:** deferred by an overwrite bug in
`_run_agent_loop` (unconditional `set_messages(initial_messages)` on
entry), fixed in exoclaw 0.20.1. After the fix, the between-
iteration heap footprint drops by roughly the size of prior history
per active subagent — the phase 2b RAM win is actually realised.

## Known gaps

### ~~2b auto-wire overwrite in `_run_agent_loop`~~ [FIXED in 0.20.1]

Earlier versions of `AgentLoop._run_agent_loop` unconditionally
called `self._executor.set_messages(initial_messages)` at the top,
wrapping `initial` in a snapshot closure that replaced the lazy
`PriorSource` `build_prompt` had just installed. So phase 2b's RAM
behaviour was structurally in place in the executor but not
observable through the real loop path.

Fixed in exoclaw 0.20.1: the seed was dropped entirely.
`initial_messages` is retained on the signature for back-compat
with monkey-patching test shims but the parameter is unused.
Production flow always seeds via `build_prompt` before the loop
body runs.

### Cgroup accounting vs Python heap accounting

Moving data from Python heap to disk doesn't always remove it from
the cgroup's memory charge. Filesystem reads populate the OS page
cache, which is accounted against the cgroup under `memory.current`
on cgroup v2. "Near-zero Python RSS" ≠ "near-zero cgroup usage"
when the process re-reads session JSONL on every iteration.

Mitigation: set `vm.dirty_ratio` appropriately and accept that page
cache will hold hot session bytes in shared kernel memory. Doesn't
save you under a cgroup limit, but the bytes are reclaimable — under
pressure the kernel can drop them without the process having a say,
which matters for the average-vs-peak story.

## Next steps toward near-zero working set

Everything below is NOT shipped. Ordered by increasing effort and
decreasing incrementality.

### ~~Step A — unblock phase 2b~~ [SHIPPED in exoclaw 0.20.1]

The `set_messages(initial_messages)` overwrite at the top of
`_run_agent_loop` was dropped. Phase 2b's lazy `PriorSource`
survives through the loop, and the between-iteration RAM floor
drops by ≈ prior history size per active subagent.

### Step B — streaming LLM request body (phase 3)

Replace `json.dumps(messages)` inside the provider adapter with a
generator that yields JSON chunks: `b'{"messages":['` → one
`json.dumps(msg)` per yielded piece → `b']}'`. Pipe into httpx's
streaming-body support. Kills the "third copy" — the JSON string
that coexists with the Python list at the moment of send.

Touches the provider protocol (currently takes `messages: list`,
needs to accept `messages: AsyncIterable | list`). Every provider
impl (LiteLLM, direct, any future) needs updating. Response path
already supports streaming (SSE) on all major providers — just need
to use it.

**Effect combined with Steps A+B:** per-subagent peak drops from the
current ~3× prompt-size (list + JSON + httpx buffer) to roughly 1×
prompt content + small streaming buffers. Memory-model-doc-original
estimate of "a few hundred KB plus unavoidable HTTP buffers" per
active subagent.

### Step C — no SessionManager cache

`SessionManager` currently holds sessions in a `WeakValueDictionary`;
as long as any caller has a strong ref, the session (with its full
`messages: list`) stays resident. Change the contract: every read
reads from disk via `mmap` or a line iterator. No session object
owns a Python list of all messages.

`DefaultConversation.load_persisted_history` becomes a generator of
fresh dicts parsed from the JSONL on demand. Hot pages end up in the
OS page cache (see "Cgroup accounting" above), not Python heap.

Touches: `SessionManager`, `DefaultConversation`, any code that
holds onto the session object expecting `.messages` to be mutable.
The in-progress turn's new messages can't live on the session
object anymore — they're the executor's delta (phase 2a already
put them there).

**Effect combined with A+B+C:** the only Python-heap state per
active subagent is the coroutine frame, the delta (messages
produced this turn), loaded skill content, and in-flight HTTP
buffers. For a turn with no tool-result explosions, that's
~50–200 KB per subagent vs. today's ~2–5 MB.

### Step D — streaming tool results

Tool results today return synchronously as a `str` held by the
Python frame until the next `_chat_step` consumes it. A 50 KB
web-fetch result, a 500 KB database query result — all held in the
message list until end of turn.

Would require a tool-protocol change to let tools stream their
output (async iterator returning chunks). The executor's tool-step
would persist chunks to a tool-result scratch file as they arrive
and pass the file handle to the next LLM call instead of the full
content. Tools that return small responses (`bool`, short strings)
stay inline; the rewrite is only for tools that legitimately produce
multi-MB results.

### Step E — workflow-frame eviction (Temporal-style)

The Temporal model: between activities, the workflow worker drops
the Python coroutine frame entirely. Next activity's completion
triggers a replay from event history up to the point where the
frame is resurrected. Per-idle-workflow RAM approaches zero.

DBOS does **not** do this today — the coroutine frame stays
resident across `@DBOS.step` await boundaries, just like a normal
asyncio coroutine. To get Temporal-style eviction on DBOS would
require upstream DBOS changes (or migrating fan-out workloads to
a Temporal executor plugin, which the `Executor` protocol
supports by design).

This is where "concurrent subagent count bounded by external
resources rather than cgroup memory" stops being a limit. The
cost is cold-start latency per activity — every resume pays the
history-replay cost. Fine for workflows that idle for seconds;
tight for sub-second tool loops.

### Step F — per-step ephemeral processes

Each step runs in a fresh subprocess: read inputs from the
journal, run the step body, write outputs, exit. No persistent
Python process per workflow. Most aggressive option on the
disk-spectrum; highest per-step latency (process fork + interpreter
start). Historically only practical for batch-style durable
executors (Airflow, Prefect); realtime agent loops would hate the
per-step overhead.

Included here for completeness — not a near-term target. If we ever
get to the point where the cgroup limit binds even with A+B+C+D
shipped, this is the next lever.

## Target numbers

Rough per-active-subagent peak memory, openclaw production turn
with ~1 MB of prompt history and ~3 LLM iterations:

| Scenario | Peak per subagent | 8 concurrent | Fits 512 MiB? |
|---|---|---|---|
| Current (exoclaw 0.20.1 — 2b realised) | ~5 MB peak, ~2 MB between iterations | ~40 MB peak, ~16 MB avg | yes, comfortably |
| Step B added | ~2 MB peak | ~16 MB | yes, with headroom |
| Steps B+C | ~200 KB peak | ~1.6 MB | trivially |
| Steps B+C+D | ~100 KB peak | ~800 KB | trivially |
| Steps B–E | ~0 (only active-step frames resident) | bounded by external queue | dropped the question |

"8 concurrent" is openclaw's current `SUBAGENT_MAX_CONCURRENT`.
With B shipped, that cap could probably move to 50–100 without
the cgroup binding. With B+C, the cap becomes an LLM
rate-limit / budget question, not a memory one.

## Practical implications

- "We have durability, so memory shouldn't matter" — still false, even
  after phase 1+2. Durability is a recovery property. RAM still
  matters for concurrent fan-out.
- Concurrent capacity per host is `total_cgroup_memory /
  per_subagent_peak`. Today ≈ 5 MB per active subagent under the
  phase 2b overwrite bug; Step A unblocking gets that structurally
  down without other code changes, but doesn't change peak (which is
  dominated by the httpx-send tripling described in "The `_messages`
  buffer" above).
- **Steps A and B are the near-term work with the biggest
  return-per-effort.** A is a one-line core fix. B is the phase 3
  streaming-httpx-body work; a few days touching the provider
  protocol. Together they drop peak enough that openclaw's 512 MiB
  cgroup stops being the binding constraint on subagent fan-out.
- Capping subagent concurrency in the spawner (see
  `SubagentSpawner.max_concurrent`) is the short-term cushion. It's
  not a bug that this cap exists; it's that the cap should be set
  by LLM rate limits and budget, not by RAM.

## When to revisit

Signals that push us to the next step:

- **Step A (unblock phase 2b)**: should be done unconditionally — it's
  a one-line fix to realise RAM reduction already in the codebase.
- **Step B (streaming request body)**: do when any single subagent's
  peak during the LLM call crosses ~10 MB (very large history, long
  tool results stacking up, or a subagent doing many iterations).
- **Step C (no SessionManager cache)**: do when a single session's
  history JSONL grows past ~5 MB and `get_or_create` holding it in
  RAM affects memory headroom across concurrent sessions.
- **Step D (streaming tool results)**: do when a tool legitimately
  needs to return multi-MB results (video transcripts, large
  document processing, bulk DB exports) and holding them in the
  message buffer starves other work.
- **Step E (workflow-frame eviction)**: do when subagent fan-out
  needs to scale past ~100 concurrent on a single host, or when a
  Temporal-executor plugin shows up and inherits this for free.
- **Step F (per-step ephemeral processes)**: do when none of the
  above are enough and per-activity latency-tolerance is high
  (think: overnight batch workflows, not interactive agents).

Until a step gets done, the pattern is: keep concurrency caps at
each layer (spawner, provider, bus), accept the current per-subagent
footprint, and treat "more RAM" as a valid answer when capping
isn't. Raising the openclaw cgroup from 512 MiB to 1 GiB is a
legitimate short-term answer that buys 2× headroom immediately;
Step A+B buy ~10× longer-term and need to land before the next big
fan-out push.
