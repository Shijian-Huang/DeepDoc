from typing import Any

from google import genai
from google.genai import errors, types


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
        try:
            response = self.client.models.generate_content(model=model, contents=prompt)
        except (errors.ClientError, errors.ServerError) as error:
            raise RuntimeError(str(error)) from error
        return response.text or ""
