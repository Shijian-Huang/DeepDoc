import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from llm.cancellation import (
    AnalysisCancelled,
    check_analysis_cancelled,
    request_cancellation_token,
)


@dataclass(frozen=True)
class OllamaProvider:
    base_url: str
    context_length: int
    num_predict: int
    keep_alive: str
    think: bool
    cpu_only: bool

    def installed_models(self, timeout: float = 1.5) -> set[str]:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return set()
        return {
            str(model.get("name") or model.get("model") or "")
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }

    def generate_text(
        self,
        prompt: str,
        model: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        check_analysis_cancelled()
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema or "json",
            "stream": True,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.1,
                "num_ctx": self.context_length,
                "num_predict": self.num_predict,
                **({"num_gpu": 0} if self.cpu_only else {}),
            },
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = None
        token = request_cancellation_token.get()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if token is not None:
                    token.attach_response(response)
                content_parts: list[str] = []
                if hasattr(response, "__iter__"):
                    for raw_line in response:
                        check_analysis_cancelled()
                        if not raw_line.strip():
                            continue
                        chunk = json.loads(raw_line.decode("utf-8"))
                        if chunk.get("error"):
                            raise RuntimeError(f"Ollama request failed: {chunk['error']}")
                        message = chunk.get("message") or {}
                        content_parts.append(str(message.get("content") or ""))
                else:
                    chunk = json.loads(response.read().decode("utf-8"))
                    message = chunk.get("message") or {}
                    content_parts.append(str(message.get("content") or ""))
                check_analysis_cancelled()
        except AnalysisCancelled:
            raise
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as error:
            if token is not None:
                token.check()
            raise RuntimeError(f"Ollama request failed: {error}") from error
        finally:
            if token is not None and response is not None:
                token.detach_response(response)

        return "".join(content_parts)
