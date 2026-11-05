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
    think: bool | str
    cpu_only: bool
    temperature: float = 0.1

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
        # GPT-OSS requires a reasoning level rather than a boolean toggle.
        think = self.think
        if model.split(":", 1)[0].split("/")[-1] == "gpt-oss" and isinstance(think, bool):
            think = "high" if think else "low"
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema or "json",
            "stream": True,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
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
                thinking_chars = 0
                final_chunk: dict[str, Any] = {}
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
                        thinking_chars += len(str(message.get("thinking") or ""))
                        if chunk.get("done"):
                            final_chunk = chunk
                else:
                    chunk = json.loads(response.read().decode("utf-8"))
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama request failed: {chunk['error']}")
                    message = chunk.get("message") or {}
                    content_parts.append(str(message.get("content") or ""))
                    thinking_chars = len(str(message.get("thinking") or ""))
                    final_chunk = chunk
                check_analysis_cancelled()
                if final_chunk.get("done_reason") == "length":
                    raise RuntimeError(
                        f"Ollama model '{model}' exhausted its output token budget "
                        f"(num_predict={self.num_predict}, eval_count={final_chunk.get('eval_count', 'unknown')}). "
                        f"Received {sum(map(len, content_parts))} answer characters and "
                        f"{thinking_chars} thinking characters. Increase the output token budget."
                    )
                if not "".join(content_parts).strip():
                    raise RuntimeError(
                        f"Ollama model '{model}' returned an empty answer "
                        f"({thinking_chars} thinking characters; "
                        f"done_reason={final_chunk.get('done_reason', 'unknown')}). "
                        "Thinking output is not a final JSON answer."
                    )
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
