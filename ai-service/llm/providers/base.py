from typing import Any, Protocol


class LLMProvider(Protocol):
    def generate_text(
        self,
        prompt: str,
        model: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        ...
