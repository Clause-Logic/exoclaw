# Plugin integration patterns

Reference for how core (`exoclaw`) connects with plugins
(`exoclaw-conversation`, `exoclaw-executor-dbos`, providers, channels,
…). These patterns aren't named in code anywhere — they're the
implicit conventions you'll find in `agent/loop.py`, `executor.py`,
`agent/conversation.py`, `providers/protocol.py`. Doc them here so the
next refactor doesn't reinvent them.

## The patterns

### 1. Required Protocol (`runtime_checkable`)

The minimal surface every implementer must provide. Lives in core,
imported by plugins.

```python
@runtime_checkable
class Conversation(Protocol):
    async def build_prompt(...) -> list[dict[str, Any]]: ...
    async def record(...) -> None: ...
    async def clear(...) -> bool: ...
    def list_sessions(...) -> list[dict[str, Any]]: ...
```

`runtime_checkable` lets `isinstance(obj, Conversation)` confirm
structural conformance at runtime — useful for tests and for narrow
type guards in dispatch code.

**Use when:** every plugin needs the method, period. Adding a new
method here is a breaking change.

### 2. Optional method, detected via `getattr`

A method that some plugins implement and some don't. Core checks for
it at the call site. **NOT on the Protocol** — that would make it
"required" by the structural type system and silently break existing
implementers.

```python
# Concrete plugin (DirectExecutor) implements:
def set_prior_source(self, src: PriorSource) -> None: ...

# Core call site:
loader = getattr(executor, "set_prior_source", None)
if loader is not None:
    loader(my_source)
```

Examples in core: `Executor.set_prior_source`, `Executor.enqueue_inbound`,
`Executor.monotonic_ms`. The convention is to add a comment block
near the protocol explaining why the method *isn't* on it.

**Use when:** adding a new method to an existing plugin surface
without breaking older implementations. Purely additive — old
plugins just don't get the new behavior.

### 3. Capability flag (`bool` attribute)

A boolean attribute the plugin sets to opt INTO a behavior change in
core. Core branches on the flag.

```python
@runtime_checkable
class Executor(Protocol):
    handles_response_send: bool
    handles_inbound_enqueue: bool
```

```python
# Core:
if getattr(executor, "handles_inbound_enqueue", False) is True:
    bus.set_inbound_hook(executor.enqueue_inbound)
```

Note the `is True` check — `MagicMock` returns truthy for any spec
attribute, so a plain truthy check would silently enable the path
for any spec'd test mock.

**Use when:** the plugin's presence-of-method isn't sufficient — core
also needs to know whether to *take a different control-flow path*.
Often paired with pattern #2 (optional method) where the flag opts
into using the optional method.

### 4. Extension protocol

A second Protocol that extends the base. Plugins implement the
extension if they support the feature; core checks via `isinstance`.

```python
@runtime_checkable
class AppendableConversation(Conversation, Protocol):
    async def append(self, session_id: str, message: dict) -> None: ...
    async def post_turn(self, session_id: str) -> None: ...

# Core:
if isinstance(conversation, AppendableConversation):
    await conversation.append(session_id, msg)
else:
    # legacy batch path
    await conversation.record(session_id, batch)
```

**Use when:** a cohesive *feature* requires multiple methods that
hang together. Better than several independent optional methods
because the static type checker enforces "if you have one, you have
them all."

### 5. Mediator (executor wraps conversation calls)

A middle layer takes the lower layer as a parameter and decides HOW
to call it. Lets durable / sandboxed / replay-safe variants wrap
each call without the lower layer having to know.

```python
@runtime_checkable
class Executor(Protocol):
    async def build_prompt(
        self,
        conversation: Conversation,
        session_id: str,
        message: str,
        ...
    ) -> list[dict[str, Any]]: ...

    async def append_message(
        self,
        conversation: Conversation,
        session_id: str,
        message: dict[str, Any],
    ) -> None: ...
```

The executor wraps the conversation method in a `@DBOS.step()` /
`@activity` / etc. — the conversation stays oblivious. Pass-through
executors (`DirectExecutor`) just forward the call inline.

**Use when:** behavior needs to be decorated by a middle layer
without changing the lower layer's API. Replay safety, durability
boundaries, sandboxing, observability spans all fit this pattern.

### 6. Sidecar state file

A plugin maintains its own per-session state in a separate file next
to the primary data (which stays append-only). Migration shim reads
legacy fields once, then ignores them.

```
sessions/
  telegram_42.jsonl                     # primary, append-only
  telegram_42.consolidation.json        # plugin state (sidecar)
```

Loader pattern:

```python
def load_state(state_dir: Path, key: str) -> State:
    sidecar = sidecar_path(state_dir, key)
    if sidecar.exists():
        return State.from_dict(json.loads(sidecar.read_text(...)))
    # Migration shim: peek at legacy primary file once, seed sidecar.
    legacy = _read_legacy_field(_legacy_path(state_dir, key))
    if legacy:
        state = State(...)
        save_state(state_dir, key, state)
        return state
    return State()
```

Atomic writes (`tmp.write` → `tmp.replace(path)`) so a crash
mid-write doesn't corrupt the sidecar.

**Use when:** a plugin needs persistent per-session state and the
primary data file (e.g. session JSONL) is, or should be, append-only.
The sidecar lets the plugin own its state lifecycle without coupling
the primary file's format to plugin internals.

### 7. Opaque hint over typed exception

When plugin-A signals plugin-B about a failure mode, prefer an opaque
identifier over a typed exception class in the shared protocol. Typed
exceptions in the shared layer become hard to remove — every renaming
or generalization is a breaking change.

```python
# AVOID — couples core to a specific failure category:
class ContextWindowExceededError(Exception): ...

# PREFER — provider attaches an opaque hint, core never interprets:
class RecoverableProviderError(Exception):
    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint  # "input_too_large", "rate_limit", ... — opaque to core
```

`ContextWindowExceededError` is the cautionary tale: a single
LLM-text-specific exception class baked permanently into the core
provider protocol. Removing or renaming it breaks every provider
that imports it.

**Use when:** cross-plugin signaling needs to be extensible and
core shouldn't grow vocabulary for things only plugins care about.

## Decision framework

| Question | Pattern |
|---|---|
| Will every plugin need this method? | Required Protocol (#1) |
| Adding a method without breaking existing impls? | Optional via `getattr` (#2) |
| Does this change how core operates? | Capability flag (#3) |
| Is this a cohesive feature set with multiple methods? | Extension protocol (#4) |
| Does a middle layer need to wrap each call (durability, replay, sandboxing)? | Mediator (#5) |
| Does a plugin need persistent state separate from primary data? | Sidecar file (#6) |
| Cross-plugin signaling that should stay extensible? | Opaque hint, not typed exception (#7) |

## Anti-patterns

- **Adding a method to a Required Protocol after release.** Breaks
  every existing implementer at the type-checker level. Use #2 or
  #4 instead.
- **Using a typed exception in a shared protocol when an opaque hint
  would do.** See `ContextWindowExceededError`.
- **Holding full session state in memory because the plugin "needs
  it."** Plugins that need session state should stream from disk
  (`SessionReader` pattern in `exoclaw-conversation`) or maintain
  their own bounded state in a sidecar (#6). Core runs on devices
  from cloud servers down to ESP32s — accidental materialization
  breaks the small end.
- **Truthy checks on capability flags.** `MagicMock(spec=Executor)`
  returns truthy for any spec attribute. Use `is True` to avoid
  silently enabling the gated path in tests.
