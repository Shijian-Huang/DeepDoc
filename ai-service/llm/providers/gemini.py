from typing import Any

from google import genai
from google.genai import errors, types

from llm.usage import set_last_usage


class GeminiProvider:
    def __init__(self, api_key: str | None, timeout_ms: int = 30000):
        self.client = (
            genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
            if api_key
            else None
        )

    def generate_text(
        self,
        prompt: str,
        model: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        del schema  # Gemini is prompted for JSON; schema enforcement is currently Ollama-only.
        if self.client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured. Add it to ai-service/.env or the server environment.")
        set_last_usage(None)
        try:
            response = self.client.models.generate_content(model=model, contents=prompt)
        except (errors.ClientError, errors.ServerError) as error:
            raise RuntimeError(str(error)) from error
        metadata = getattr(response, "usage_metadata", None)
        if metadata is not None:
            set_last_usage({
                "prompt_tokens": getattr(metadata, "prompt_token_count", None),
                "output_tokens": getattr(metadata, "candidates_token_count", None),
                "thinking_tokens": getattr(metadata, "thoughts_token_count", None),
            })
        return response.text or ""
