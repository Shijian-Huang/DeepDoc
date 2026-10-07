"""Summary-model interfaces and the models exposed by the evaluator dashboard.

Add or remove entries in ``EVALUATION_MODELS`` to change the dashboard without
touching its UI or orchestration code.  Every entry implements the callable
interface accepted by :class:`llm.evaluator.SummaryEvaluator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from llm import summarizer


SummaryValue = str | Mapping[str, Any]


class EvaluationModel(Protocol):
    """The model contract consumed by the evaluation dashboard."""

    name: str

    def summarize(self, prompt: str) -> SummaryValue:
        """Generate a summary response from a fully constructed prompt."""


@dataclass(frozen=True)
class ConfiguredLLMModel:
    """Adapter for one of DeepDoc's configured Gemini or Ollama models."""

    name: str
    provider: str
    model: str

    def summarize(self, prompt: str) -> SummaryValue:
        with summarizer.use_llm_selection(self.provider, self.model):
            return summarizer.generate_json(prompt)


# This is intentionally explicit: it is the single source of truth for the
# model names shown in the evaluator dashboard.
EVALUATION_MODELS: tuple[EvaluationModel, ...] = (
    # ConfiguredLLMModel("Qwen3 4B", "ollama", "qwen3:4b"),
    # ConfiguredLLMModel("Qwen3 8B", "ollama", "qwen3:8b"),
    # ConfiguredLLMModel("Qwen3 14B", "ollama", "qwen3:14b"),
    ConfiguredLLMModel(
        "Gemini 3.1 Flash Lite Preview",
        "gemini",
        "gemini-3.1-flash-lite-preview",
    ),
    ConfiguredLLMModel("Gemini 3.5 Flash Lite", "gemini", "gemini-3.5-flash-lite"),
)


def model_registry(
    models: tuple[EvaluationModel, ...] = EVALUATION_MODELS,
) -> dict[str, EvaluationModel]:
    """Return the validated, display-name keyed model registry."""
    registry = {model.name: model for model in models}
    if len(registry) != len(models) or any(not name.strip() for name in registry):
        raise ValueError("Evaluation model names must be non-empty and unique.")
    return registry

