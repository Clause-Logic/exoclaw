# Contributing to exoclaw

Thanks for contributing! This doc covers the conventions we care about. The codebase is small enough to read in an afternoon — these guidelines keep it that way.

## Setup

```bash
uv sync --dev
```

## Quality Commands

All commands are available as mise tasks:

| Command | What it does |
|---------|-------------|
| `mise run test` | Run tests with coverage (95% minimum) |
| `mise run lint` | Type check with `ty` + lint with `ruff` |
| `mise run format` | Format code with `ruff` |
| `mise run check-format` | Check formatting without writing changes |

Or run directly:

```bash
uv run pytest --cov=exoclaw --cov-fail-under=95 tests/
uv run ty check exoclaw
uv run ruff check exoclaw tests
uv run ruff format exoclaw tests
```

CI runs `lint`, `check-format`, and `test` on every PR.

## Code Style

- `ruff format` and `ruff check` must pass — config is in `pyproject.toml`
- `ty check` for type checking — no `# type: ignore` without a comment explaining why
- Ruff rules: `E`, `F`, `I`, `N`, `W`, `TID251`, `ANN` enabled

## Logging

exoclaw uses `structlog`. All log output should follow these conventions.

### One log per action

Each meaningful action gets exactly one log line. That line should be self-contained — a reader should understand what happened without finding other log lines.

```python
# GOOD — one wide log at completion
logger.info("tool_call", **{"tool.name": name, "duration_ms": dur_ms, "result.length": len(result)})

# AVOID — scattered before/after pair
logger.info("tool_call_started", **{"tool.name": name})
# ... do work ...
logger.info("tool_call_completed", **{"tool.name": name, "duration_ms": dur_ms})
```

### Event-like messages

Short, structured, present-tense action verbs. Under 4 words.

```python
# GOOD
logger.info("tool_call", ...)
logger.info("context_compact", ...)
logger.info("channel_start_error", ...)

# AVOID
logger.info("Starting to execute the tool", ...)
logger.info("Successfully compacted context", ...)
```

### Bind context early, log lean

Bind stable identifiers once at the top of a function or activity. Then individual log calls only add ephemeral data.

```python
# Bind once
structlog.contextvars.bind_contextvars(**{"session.key": session_key, "chat.id": chat_id})

# Log lean — context is already there
logger.info("tool_call", **{"tool.name": name, "duration_ms": dur_ms})
```

### Attribute keys follow OTel conventions

Use lowercase, dot-namespaced keys per [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/general/naming/). Since dots aren't valid Python keyword arguments, use `**{}` to pass them.

```python
# GOOD — OTel-style dot-namespaced keys
logger.info("sandbox_exec", **{"process.exit_code": rc, "result.length": len(out)})

# AVOID — camelCase or flat keys
logger.info("sandbox_exec", exitCode=rc, resultLen=len(out))
```

Use standard OTel names when they exist (`http.method`, `http.status_code`, `db.system`). App-specific keys get a domain prefix (`sandbox.intent`, `bulk_qa.question_count`).

### No string interpolation

Structure goes in key-value pairs, not in the message.

```python
# GOOD
logger.info("channel_start_error", **{"channel.name": name, "error": str(e)})

# AVOID
logger.info(f"Failed to start channel {name}: {e}")
```

## Tests

- All new code needs tests
- `mise run test` must pass before submitting a PR
- Coverage target: 95% — enforced by `--cov-fail-under`

## Pull Requests

- Keep PRs focused — one concern per PR
- The core (`agent/loop.py`, protocols, registry) changes rarely by design. If your change touches the core, explain why in the PR description.
- New tools, channels, and providers should live in separate plugin packages, not in this repo
