"""LLM-provider protocols used by the core execution layer."""

from typing import Awaitable, Callable, Protocol, runtime_checkable

from exoclaw.providers.types import LLMResponse, ResponseFormat


@runtime_checkable
class LLMProvider(Protocol):
    """Baseline provider surface.

    Keep this protocol stable: providers are structurally typed, so adding a
    keyword here makes existing provider implementations fail type checks even
    when the new capability is unused. Optional streaming lives in the
    separate ``StreamingLLMProvider`` capability below.
    """

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse: ...

    def get_default_model(self) -> str: ...


class StreamingLLMProvider(LLMProvider, Protocol):
    """Optional provider capability for visible content-delta streaming."""

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        response_format: ResponseFormat | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse: ...
