"""LLMProvider protocol — the only provider surface core depends on."""

from typing import Protocol, runtime_checkable

from exoclaw.providers.types import LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse: ...

    def get_default_model(self) -> str: ...
