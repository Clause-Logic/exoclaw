"""Shared data types for LLM provider responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Required, TypedDict, Union


class ContextWindowExceededError(Exception):
    """Raised by providers when the prompt exceeds the model's context window."""


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


# ---------------------------------------------------------------------------
# Response format types (compatible with OpenAI SDK types)
# ---------------------------------------------------------------------------


class ResponseFormatText(TypedDict, total=False):
    """Default response format. Used to generate text responses."""

    type: Required[str]  # Literal["text"]


class JSONSchema(TypedDict, total=False):
    """Structured Outputs configuration options, including a JSON Schema."""

    name: Required[str]
    """Must be a-z, A-Z, 0-9, underscores and dashes, max 64 chars."""

    description: str
    """Description of what the response format is for."""

    schema: dict[str, object]
    """The schema for the response format, as a JSON Schema object."""

    strict: Optional[bool]
    """Whether to enable strict schema adherence."""


class ResponseFormatJSONSchema(TypedDict, total=False):
    """JSON Schema response format for structured outputs."""

    json_schema: Required[JSONSchema]
    """Structured Outputs configuration options, including a JSON Schema."""

    type: Required[str]  # Literal["json_schema"]


class ResponseFormatJSONObject(TypedDict, total=False):
    """JSON object response format. Prefer json_schema for models that support it."""

    type: Required[str]  # Literal["json_object"]


ResponseFormat = Union[ResponseFormatText, ResponseFormatJSONSchema, ResponseFormatJSONObject]
