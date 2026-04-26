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

### ~~Step B — streaming LLM request body~~ [SHIPPED]

*exoclaw-provider-openai 0.1.0 (PR exoclaw-plugins#60)*

`exoclaw-provider-openai._stream_body` emits the request body as
`AsyncIterable[bytes]` into `httpx.AsyncClient.post(..., content=...)`.
The body head (model, temperature, tools, stream_options) serializes
once; then one `json.dumps(msg)` per message, yielded into the socket
write. Each message's bytes can be GC'd before the next one is
serialized. The "third copy" of the prompt list — the contiguous JSON
string that used to coexist with the Python list at the moment of send
— never exists.

`LiteLLMProvider` was not migrated to this shape; its callers still
pay the snapshot cost. The deployed openclaw turn path uses
`provider-openai`, so Step B is realised in production.

### ~~Step C — no SessionManager cache~~ [SHIPPED, opt-in]

*exoclaw-conversation 0.18.0 · exoclaw-nanobot 0.25.0 · exoclaw 0.23.0*

`SessionManager(streaming_history=True)` no longer populates
`session.messages` from JSONL on `_load`. The unconsolidated tail
lives only on disk; `read_history(key, max_messages=N)` reads it
on demand via `load_range`. `_prepare_turn` skips
`session.messages.append` under streaming so the in-memory list
stays empty across turns.

The `HistoryStore` Protocol grew an additive `read_history` method
with a default impl that materializes via `session.get_history` —
non-streaming backends keep working unchanged. DB-backed backends
implement `read_history` (and `load_range`) via cursor.

`MemoryStore` takes an optional `history` ref so legacy
`consolidate()` and the boundary-repair pass in `consolidate_messages`
fall back to `load_range` when `session.messages` is empty.
`SummarizingConsolidationPolicy.should_consolidate` switched to
`session.total_messages` so the consolidation trigger is correct
under streaming.

`DirectExecutor.build_prompt` learned the same lazy-`PriorSource`
auto-wire `DBOSExecutor` had since 0.13.1, so the win is realised
on either executor backend.

`nanobot.app.create()` constructs `SessionManager(streaming_history=True)`
by default, so the deployed bot gets the win automatically. The flag
defaults `False` at the `SessionManager` constructor level for
back-compat with five `TestSessionManager` tests that read
`session.messages` after a fresh `_load`.

**Empirical (from `packages/exoclaw-nanobot/scripts/`):**
- `memory_loadtest.py`: single chat × 400 turns, tracemalloc current
  is dead flat at ~600 KiB across all 400 turns under streaming;
  cached climbs linearly to ~1056 KiB.
- `memory_loadtest_concurrent.py`: at N=64 concurrent in-flight chats
  with 100 turns of pre-seeded history each, post-turn baseline is
  ~10,735 KiB cached vs ~783 KiB streaming. ~13.7× reduction.

The in-flight peak (during `chat()`) is unchanged by Step C alone —
that's Step B territory and was already shipped.

**Follow-up not covered:** flip the default to `True` and delete the
`False` branch + the five `TestSessionManager` tests it props up,
once a soak window confirms no regressions. ~30-line cleanup.

### Step D — streaming tool results [REMAINING]

With A+B+C shipped, **Step D is the last load-bearing memory-model
item.** Tool results today return synchronously as a `str` held by
the Python frame until the next `_chat_step` consumes it. A 50 KB
web-fetch, a 500 KB database query, a multi-MB document read — all
held in the message list at full size until end of turn. On a
multi-tenant cgroup, a single fat tool result on one session can
crowd out idle sessions across the host. On a small-target single-
tenant deployment (see "Aspirational target" below), it can blow
the box.

The fix: tool-protocol change to let tools stream their output (async
iterator returning chunks). The executor's tool-step persists chunks
to a per-tool-call scratch file as they arrive and passes a file
handle to the next LLM call instead of the full content. Step B's
`AsyncIterable[bytes]` request body already accepts file-backed
input — so the executor → provider piping is already in place; what's
missing is the tool-side output protocol and the executor's
tool-step disk-write.

Tools that return small responses (`bool`, short strings, status
messages) stay inline. The rewrite is only for tools that
legitimately produce multi-MB results: web-fetch, exec stdout,
database queries, file reads, MCP tool returns over a certain
threshold. Most tools fall on the "small" side and don't change.

Approximate effort: 1-2 weeks. Touches: tool protocol (`Tool.execute`
becomes optionally async-iterable), `Executor.execute_tool`
implementations on `DirectExecutor` + `DBOSExecutor`, the providers'
message-payload assembly to read from file when given a handle, and
the few large-output tools (`WebFetchTool`, `ExecTool`, MCP wrapper).

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
| Pre-2b (exoclaw <0.20.1) | ~5 MB peak, ~2 MB between iterations | ~40 MB peak, ~16 MB avg | yes, comfortably |
| ~~Step B added~~ [SHIPPED] | ~2 MB peak | ~16 MB | yes, with headroom |
| ~~Steps B+C~~ [SHIPPED — current state] | ~200 KB peak, ~constant between turns | ~1.6 MB peak; **post-turn baseline ~constant in N** | trivially |
| Steps B+C+D | ~100 KB peak (no fat tool result inflation) | ~800 KB | trivially |
| Steps B+C+D + dep prune | minimum-CPython floor (~25-30 MiB process RSS for 1+ sessions) | n/a — bounded by N-independent cost | trivially |

"8 concurrent" is openclaw's `SUBAGENT_MAX_CONCURRENT` cap. With
B+C shipped (the current production state), the cap is an LLM
rate-limit / budget question, not a memory one. The empirical
post-turn baseline is now ~constant per host in N.

## Aspirational target — the small-board litmus test

Useful even though we don't sell on it: **can a single-tenant
nanobot run on a board you'd hold in one hand?** A coherent
yes-answer means the memory work is done in the meaningful sense.

Two reference targets, in order of decreasing ambition:

- **Pi Zero 2 W (512 MiB RAM, ~$15) — CPython.** With Steps B+C
  shipped (today) plus a dep prune (see below), single-tenant
  nanobot fits comfortably. ~30-50 MiB RSS for the agent, plenty
  of headroom for httpx / TLS / OS. Probably reachable now once
  someone tests it.

- **ESP32-S3 + 8 MiB PSRAM, MicroPython.** This is the real
  stretch goal. CPython's interpreter floor (~12-15 MiB just for
  imports) makes 8 MiB unreachable inside CPython. The aspirational
  checkpoint is: **could the same protocols, with different
  implementations of the bottom layer, run on MicroPython?**

What the MicroPython checkpoint requires (none of it research, all
of it portability work):

1. **Step D shipped** so a single fat tool result can't blow the box.
2. **Dep prune** — make `pydantic`, `structlog`, `loguru`, `litellm`,
   `pydantic-settings`, `dbos` truly optional via `try: import` at
   the call site, not at module load. The deployed bot doesn't need
   any of them strictly; nanobot's config layer is the only hard
   `pydantic` user. Move config to `TypedDict` + a 50-line validator.
   ~5 days.
3. **Concurrency-isolation shim** — `asyncio.ContextVar` is
   load-bearing for prior/delta isolation, structlog binding, DBOS
   context. Single-tenant single-flight target doesn't need this;
   add a `runtime.task_local()` shim that's `ContextVar` on CPython,
   plain global on micro. ~½ day.
4. **`MicroOpenAIProvider`** — new tiny package, ~200-300 lines.
   `usocket` + `ussl` + hand-rolled SSE parser. Strict subset of
   `exoclaw-provider-openai`'s `_stream_body` shape (which already
   yields one message at a time, perfect for the micro target).
   No router, no fallback chain, no cache-control stamping. ~3 days.
5. **Drop `weakref.WeakValueDictionary`** from `SessionManager`
   under streaming (the cache is nearly useless after Step C
   anyway). LRU dict or no cache at all. ~½ day.
6. **`DirectExecutor` only** — MicroPython has no `DBOS`, no
   Postgres, no threading model that matters. Drop durability for
   the micro target; rely on the JSONL flush already in phase 1.

Total porting cost above: roughly 2 focused weeks once Step D is
done. The exoclaw `Executor` and `LLMProvider` protocols are
already designed for this — `DirectExecutor` and `DBOSExecutor`
coexist today, `LiteLLMProvider` and `OpenAIProvider` coexist
today. A `MicroOpenAIProvider` is just a third sibling at the
provider layer.

**Why this is a useful target even if we never deploy it:** every
item on the list is also a memory or portability win for the
production-server case. Step D bounds tool-result blow-ups under
multi-tenant pressure. The dep prune lets us boot faster and use
less RAM at idle. The `task_local` shim makes the isolation story
explicit instead of implicit. The micro provider is a leaner
fallback when LiteLLM is unwanted. None of this work is wasted on
the production deployment.

The implicit checkpoint: **once Step D ships and a Pi Zero 2 W
demo works, the headline memory work is done.** ESP32-S3 / micro
is a follow-up reachable from there with portability discipline.

## Practical implications

- "We have durability, so memory shouldn't matter" — still false, even
  after phase 1+2. Durability is a recovery property. RAM still
  matters for concurrent fan-out.
- Concurrent capacity per host is `total_cgroup_memory /
  per_subagent_peak`. With A+B+C shipped (current state), the
  per-session post-turn baseline is ~constant in session length and
  ~constant in N concurrent sessions. The cap on host capacity is
  now in-flight peak — bounded by Step D for tool-result blowouts.
- **Step D is the remaining near-term work** with meaningful
  return-per-effort: bounds tool-result inflation across the
  multi-tenant turn path, and is the prerequisite for the
  small-board aspirational target.
- Capping subagent concurrency in the spawner (see
  `SubagentSpawner.max_concurrent`) is no longer the load-bearing
  cushion it was — with B+C shipped the cap is an LLM rate-limit
  / budget question, not a memory one. Keep it set by those
  considerations, not RAM headroom.

## When to revisit

Signals that push us to the next step:

- ~~**Step A (unblock phase 2b)**~~ — shipped, exoclaw 0.20.1.
- ~~**Step B (streaming request body)**~~ — shipped via
  `exoclaw-provider-openai`. The deployed openclaw turn path
  uses it.
- ~~**Step C (no SessionManager cache)**~~ — shipped opt-in
  (`streaming_history=True`); on by default in `nanobot.app`.
  Empirically: ~13.7× post-turn baseline reduction at N=64
  concurrent sessions.
- **Step D (streaming tool results)**: do when a tool legitimately
  needs to return multi-MB results (video transcripts, large
  document processing, bulk DB exports) and holding them in the
  message buffer starves other work — *or* when the small-board
  target becomes a serious goal. With A+B+C shipped, D is the
  load-bearing remaining piece.
- **Step E (workflow-frame eviction)**: do when subagent fan-out
  needs to scale past ~100 concurrent on a single host, or when a
  Temporal-executor plugin shows up and inherits this for free.
  Not on the small-board path — single-tenant has nothing to evict.
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
