# Conversation compaction

This document surveys how conversation compaction and memory consolidation work across four exoclaw-family implementations, and identifies the best ideas worth adopting.

---

## How each implementation handles it

### exoclaw-conversation (Python, current)

Two-layer memory: `MEMORY.md` (long-term facts) + `HISTORY.md` (grep-searchable log).

**Consolidation flow** (`memory.py: MemoryStore.consolidate()`):
1. Takes messages older than `memory_window` (default 50), keeps the last 25.
2. Formats them as a timestamped transcript.
3. Calls the LLM with a `save_memory` tool, passing current `MEMORY.md` + old messages.
4. LLM produces two outputs via tool call:
   - `history_entry` — 2-5 sentence paragraph appended to `HISTORY.md`
   - `memory_update` — full updated `MEMORY.md`
5. Advances `session.last_consolidated` so the same messages aren't reprocessed.

Old messages stay in the JSONL session file — they are summarized, not deleted. `archive_all=True` processes the entire session at once (used on session end).

**Gaps:**
- No emergency fallback: if the LLM doesn't call `save_memory`, `consolidate()` returns `False` and nothing happens.
- No forced truncation path when the context limit is hit mid-session.
- No multi-part summarization for very large histories.

---

### openclaw-src (TypeScript)

Tree-structured JSONL transcripts (entries have `id` + `parentId`) with two-layer persistence.

**Auto-compaction triggers:**
- **Overflow recovery**: model returns a context overflow error → compact → retry.
- **Threshold**: `contextTokens > contextWindow - reserveTokens` (default headroom: 16 384 tokens).

**Compaction entries** persisted in the transcript: `compaction` (with `firstKeptEntryId`, `tokensBefore`) and `branch_summary` (when navigating tree branches).

**Pre-compaction memory flush:** Before compacting, runs a silent agentic turn to write durable state to disk. Uses a `NO_REPLY` convention so the output is invisible to the user. Tracked in the session store via `memoryFlushAt` and `memoryFlushCompactionCount`.

**Settings:** `compaction.reserveTokens`, `compaction.keepRecentTokens` (default 20 000), `compaction.reserveTokensFloor` (default 20 000).

---

### picoclaw (Go)

File-backed markdown memory: `MEMORY.md` + daily notes at `memory/YYYYMM/YYYYMMDD.md`.

**Threshold trigger** (`loop.go: maybeSummarize()`): history > 20 messages OR token estimate > 75% of context window.

**Multi-part summarization** (`loop.go: summarizeSession()`): For large histories (> 10 messages), splits into 2 batches, summarizes each separately, then merges. Keeps the last 4 messages for continuity.

**Oversized message guard:** Skips messages > 50% of context window and notes the omission in the summary.

**Emergency fallback** (`forceCompression()`): Drops the oldest 50% of messages, appends a compression note to the system prompt. Triggered when the limit is hit and normal summarization has already run.

**Context caching:** Static system prompt (identity, bootstrap files, skills, memory) cached and auto-invalidated by mtime. Only the dynamic parts (time, session summary) are rebuilt per turn.

---

### zeroclaw (Python)

No compaction. LangGraph-based stateless agent with simple key-value memory tools (`memory_store(key, value)` / `memory_recall(query)` backed by `~/.zeroclaw/memory_store.json`). No summarization, no context window management.

---

## Comparison

| | exoclaw-conversation | openclaw-src | picoclaw | zeroclaw |
|---|---|---|---|---|
| Memory format | MEMORY.md + HISTORY.md | Tree JSONL | MEMORY.md + daily notes | JSON key-value |
| Compaction trigger | Manual / on session end | Overflow + threshold | Threshold (20 msgs / 75%) | None |
| Summarization | LLM tool call | Pi runtime | Multi-part LLM batches | None |
| Emergency fallback | None | Overflow recovery | Drop oldest 50% | N/A |
| Pre-compaction flush | No | Yes (silent agentic turn) | No | N/A |
| Context caching | No | No | mtime-based static cache | N/A |

---

## Best ideas to adopt

**From picoclaw:**
- **Multi-part summarization**: Split large histories into batches before summarizing. Prevents single-call token overflow during the consolidation itself.
- **Emergency fallback**: When consolidation fails or the context limit is already hit, drop the oldest N% of messages rather than failing the turn entirely. Append a note to the system prompt so the LLM knows.
- **Threshold-based trigger**: Don't wait for session end — consolidate proactively when message count or estimated token count crosses a threshold.

**From openclaw-src:**
- **Pre-compaction memory flush**: Before compacting, run a silent turn whose only job is to persist important state to disk. This ensures nothing load-bearing is lost in the summarization. Suppress its output with a `NO_REPLY` convention or equivalent.
- **Configurable token headroom**: Reserve a fixed buffer (`reserveTokens`) so there is always room for the next user turn and LLM output after compaction.
