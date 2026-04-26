"""Shared data types for LLM provider responses.

``ToolCallRequest`` and ``LLMResponse`` are plain classes with
explicit ``__init__`` rather than ``@dataclass`` — MicroPython 1.27
strips ``name: type`` annotations at compile time, so a runtime
dataclass decorator can't introspect them. Manually-written
``__init__`` is the cross-runtime path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Required, TypedDict, Union


class ContextWindowExceededError(Exception):
    """Raised by providers when the prompt exceeds the model's context window."""


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


class ResponseFormatText(TypedDict):
    """Default response format. Used to generate text responses."""

    type: Required[str]  # Literal["text"]


class JSONSchema(TypedDict):
    """Structured Outputs configuration options, including a JSON Schema."""

    name: Required[str]
    """Must be a-z, A-Z, 0-9, underscores and dashes, max 64 chars."""

    description: str
    """Description of what the response format is for."""

    schema: dict[str, object]
    """The schema for the response format, as a JSON Schema object."""

    strict: Optional[bool]
    """Whether to enable strict schema adherence."""


class ResponseFormatJSONSchema(TypedDict):
    """JSON Schema response format for structured outputs."""

    json_schema: Required[JSONSchema]
    """Structured Outputs configuration options, including a JSON Schema."""

    type: Required[str]  # Literal["json_schema"]


class ResponseFormatJSONObject(TypedDict):
    """JSON object response format. Prefer json_schema for models that support it."""

    type: Required[str]  # Literal["json_object"]


# ``Union[A, B, C]`` is a runtime subscription — MicroPython 1.27's
# typing stub doesn't carry full Union machinery, so gate the alias
# on ``TYPE_CHECKING`` (annotations are strings under ``__future__``
# so the runtime placeholder is unused).
if TYPE_CHECKING:
    ResponseFormat = Union[ResponseFormatText, ResponseFormatJSONSchema, ResponseFormatJSONObject]
else:
    ResponseFormat = object
