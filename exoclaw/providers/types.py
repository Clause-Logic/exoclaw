"""Shared data types for LLM provider responses.

CPython gets real ``@dataclass`` so downstream callers (executor
plugins, nanobot config layer, anything that uses
``dataclasses.asdict`` / ``fields`` / ``is_dataclass`` for journaling
or serialization) keep working unchanged. MicroPython gets the same
classes as plain classes with a hand-written ``__init__`` because
MP strips ``name: type`` annotations at compile time, so a runtime
``@dataclass`` decorator can't introspect them.

Same constructor signatures + same attribute shape on both
runtimes — only difference is ``@dataclass`` machinery (asdict,
__eq__, __repr__) is CPython-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, Optional, Required, TypedDict, Union

from exoclaw._compat import IS_MICROPYTHON


class ContextWindowExceededError(Exception):
    """Raised by providers when the prompt exceeds the model's context window."""


if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass, field

    @dataclass
    class ToolCallRequest:
        """A tool call request from the LLM."""

        id: str
        name: str
        arguments: dict[str, object]

    @dataclass
    class LLMResponse:
        """Response from an LLM provider."""

        content: str | None
        tool_calls: list[ToolCallRequest] = field(default_factory=list)
        finish_reason: str = "stop"
        usage: dict[str, int] = field(default_factory=dict)
        reasoning_content: str | None = None
        thinking_blocks: list[dict[str, object]] | None = None

        @property
        def has_tool_calls(self) -> bool:
            return len(self.tool_calls) > 0

else:  # pragma: no cover (cpython)

    class ToolCallRequest:
        """A tool call request from the LLM."""

        def __init__(self, id: str, name: str, arguments: dict[str, object]) -> None:
            self.id = id
            self.name = name
            self.arguments = arguments

    class LLMResponse:
        """Response from an LLM provider."""

        def __init__(
            self,
            content: str | None,
            tool_calls: list[ToolCallRequest] | None = None,
            finish_reason: str = "stop",
            usage: dict[str, int] | None = None,
            reasoning_content: str | None = None,
            thinking_blocks: list[dict[str, object]] | None = None,
        ) -> None:
            self.content = content
            self.tool_calls = tool_calls if tool_calls is not None else []
            self.finish_reason = finish_reason
            self.usage = usage if usage is not None else {}
            self.reasoning_content = reasoning_content
            self.thinking_blocks = thinking_blocks

        @property
        def has_tool_calls(self) -> bool:
            return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Response format types (compatible with OpenAI SDK types)
# ---------------------------------------------------------------------------
#
# The TypedDict subclasses below carry only field annotations.
# CPython evaluates annotations at class-creation time and counts
# them as covered; MicroPython compiles annotations away entirely,
# so coverage.py's executable line set includes lines that the MP
# bytecode never actually executes. Mark each ``name: Type``
# annotation as MP-excluded so the matrix entry isn't penalised
# for the asymmetry — the runtime impact is zero either way.


class ResponseFormatText(TypedDict):
    """Default response format. Used to generate text responses."""

    type: Required[str]  # Literal["text"] # pragma: no cover (micropython)


class JSONSchema(TypedDict):
    """Structured Outputs configuration options, including a JSON Schema.

    The ``total=False`` class kwarg this used to carry can't appear
    on MicroPython (1.27 doesn't accept class kwargs), so per-field
    ``NotRequired[...]`` markers stand in. Same type-checker
    semantics; works on both runtimes."""

    name: Required[str]  # pragma: no cover (micropython)
    """Must be a-z, A-Z, 0-9, underscores and dashes, max 64 chars."""  # pragma: no cover (micropython)

    description: NotRequired[str]  # pragma: no cover (micropython)
    """Description of what the response format is for."""  # pragma: no cover (micropython)

    schema: NotRequired[dict[str, object]]  # pragma: no cover (micropython)
    """The schema for the response format, as a JSON Schema object."""  # pragma: no cover (micropython)

    strict: NotRequired[Optional[bool]]  # pragma: no cover (micropython)
    """Whether to enable strict schema adherence."""  # pragma: no cover (micropython)


class ResponseFormatJSONSchema(TypedDict):
    """JSON Schema response format for structured outputs."""

    json_schema: Required[JSONSchema]  # pragma: no cover (micropython)
    """Structured Outputs configuration options, including a JSON Schema."""  # pragma: no cover (micropython)

    type: Required[str]  # Literal["json_schema"] # pragma: no cover (micropython)


class ResponseFormatJSONObject(TypedDict):
    """JSON object response format. Prefer json_schema for models that support it."""

    type: Required[str]  # Literal["json_object"] # pragma: no cover (micropython)


# ``Union[A, B, C]`` is a runtime subscription — MicroPython 1.27's
# typing stub doesn't carry full Union machinery, so gate the alias
# on ``TYPE_CHECKING`` (annotations are strings under ``__future__``
# so the runtime placeholder is unused).
if TYPE_CHECKING:
    ResponseFormat = Union[ResponseFormatText, ResponseFormatJSONSchema, ResponseFormatJSONObject]
else:
    ResponseFormat = object
