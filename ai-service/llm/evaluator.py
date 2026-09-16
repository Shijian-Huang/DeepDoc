from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from utils.summary_prompt import build_research_summary_prompt, normalize_summary_mode

DEFAULT_PACKETS_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "default_packets.json"
DEFAULT_EVALUATION_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


class EvaluationError(RuntimeError):
    """Raised when an evaluation input or judge response is invalid."""


@dataclass(frozen=True)
class EvidencePacket:
    id: str
    text: str


class EvaluationJudge(Protocol):
    """Minimal interface implemented by any evaluator/judge model."""

    def evaluate(self, summary: str, evidence_packet: EvidencePacket) -> Mapping[str, Any]:
        ...


SummaryValue = str | Mapping[str, Any]
SummaryProvider = SummaryValue | Callable[[str], SummaryValue]
SummaryGenerator = Callable[[str, str], SummaryValue]

SCORE_FIELDS = (
    "avg_score",
    "key_ideas_score",
    "contributions_score",
    "hallucination_score",
)


class GeminiJudge:
    """Gemini-backed judge; replace this class through dependency injection."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_EVALUATOR_MODEL", "gemini-3.5-flash-lite")
        self._api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise EvaluationError(
                "GEMINI_API_KEY is not configured for the evaluation judge."
            )
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=30000),
        )
        return self._client

    def evaluate(self, summary: str, evidence_packet: EvidencePacket) -> Mapping[str, Any]:
        client = self._get_client()
        prompt = build_evaluation_prompt(summary, evidence_packet)
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
        except Exception as error:
            raise EvaluationError(f"Gemini judge request failed: {error}") from error

        raw_text = str(response.text or "").strip()
        try:
            return json.loads(_extract_json(raw_text))
        except (json.JSONDecodeError, TypeError) as error:
            raise EvaluationError("Gemini judge did not return valid JSON.") from error


class SummaryEvaluator:
    """Evaluate every model summary against every selected evidence packet.

    ``models`` is normally a mapping from model name to either a generated
    summary or a callable that generates a summary from the shared prompt. A
    model name (or sequence of names) can instead be supplied with
    ``summary_generator``.
    """

    def __init__(
        self,
        models: str | Sequence[str] | Mapping[str, SummaryProvider],
        evidence_packets: Sequence[str | Mapping[str, Any] | EvidencePacket] | None = None,
        *,
        judge: EvaluationJudge | None = None,
        summary_generator: SummaryGenerator | None = None,
        summary_mode: str = "standard",
        default_packets_path: str | Path | None = None,
        evaluation_dir: str | Path | None = None,
    ) -> None:
        self._model_sources = _normalize_models(models, summary_generator)
        self._judge = judge or GeminiJudge()
        self._summary_generator = summary_generator
        self._summary_mode = normalize_summary_mode(summary_mode)
        self._default_packets_path = Path(default_packets_path or DEFAULT_PACKETS_PATH)
        self._evaluation_dir = Path(evaluation_dir or DEFAULT_EVALUATION_DIR)
        self._packets = (
            _normalize_packets(evidence_packets)
            if evidence_packets is not None
            else load_default_evidence_packets(self._default_packets_path)
        )
        if not self._packets:
            raise EvaluationError("At least one evidence packet is required.")

    def evaluate(self, mode: str = "default", *, visualize: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible result for all model/packet combinations."""
        if mode not in {"default", "detailed", "average"}:
            raise EvaluationError("mode must be 'default', 'detailed', or 'average'.")
        if visualize and mode != "average":
            raise EvaluationError("Visualization is only available in average mode.")

        packet_results: list[dict[str, Any]] = []
        for model_name, source in self._model_sources.items():
            for packet in self._packets:
                summary = self._resolve_summary(model_name, source, packet)
                judgment = _normalize_judgment(self._judge.evaluate(summary, packet))
                evaluation = (
                    _detailed_result(judgment)
                    if mode == "detailed"
                    else _score_result(judgment)
                )
                packet_results.append({
                    "model": model_name,
                    "packet_id": packet.id,
                    "evaluation": evaluation,
                })

        if mode != "average":
            return _group_packet_results_by_model(packet_results)

        average_results = _average_results(packet_results)
        result: dict[str, Any] = {"mode": mode, "results": average_results}
        if visualize:
            chart_path = save_average_scores_chart(
                average_results,
                output_dir=self._evaluation_dir,
            )
            result["visualization_path"] = str(chart_path)
        return result

    def evaluate_json(
        self,
        mode: str = "default",
        *,
        visualize: bool = False,
        indent: int | None = None,
    ) -> str:
        """Return the evaluation as serialized, standards-compliant JSON."""
        return json.dumps(
            self.evaluate(mode, visualize=visualize),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def _resolve_summary(
        self,
        model_name: str,
        source: SummaryProvider | None,
        packet: EvidencePacket,
    ) -> str:
        prompt = build_research_summary_prompt(packet.text, self._summary_mode)
        if callable(source):
            value = source(prompt)
        elif source is not None:
            value = source
        elif self._summary_generator is not None:
            value = self._summary_generator(model_name, prompt)
        else:
            raise EvaluationError(
                f"No summary or summary_generator was supplied for model '{model_name}'."
            )
        return _summary_text(value)


def load_default_evidence_packets(path: str | Path = DEFAULT_PACKETS_PATH) -> list[EvidencePacket]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Could not load default evidence packets from {path}.") from error
    if not isinstance(payload, list):
        raise EvaluationError("The default evidence packet file must contain a JSON array.")
    return _normalize_packets(payload)


def build_evaluation_prompt(summary: str, packet: EvidencePacket) -> str:
    return f"""
You are a strict evaluator of a research-paper summary. The evidence packet is
the only source of truth. Do not use outside knowledge.

First identify the distinct key ideas and distinct novel contributions that a
faithful summary should contain. Then compare the candidate summary with them.
Mark an item covered only when its meaning is represented accurately, partial
when it is present but materially incomplete, and missing otherwise. Identify
every externally verifiable claim in the summary that is unsupported by or
contradicts this packet. Do not count harmless wording differences as errors.

Return ONLY valid JSON with exactly this internal assessment shape:
{{
  "key_ideas": {{
    "covered": 0,
    "partial_covered": 0,
    "expected": 0,
    "missing": ["description of each missing idea"]
  }},
  "contributions": {{
    "covered": 0,
    "partial_covered": 0,
    "expected": 0,
    "missing": ["description of each missing contribution"]
  }},
  "hallucinated_claims": [
    {{"claim": "...", "reason": "...", "evidence": null}}
  ]
}}

Packet id: {packet.id}
Evidence packet:
{packet.text}

Candidate summary:
{summary}
""".strip()


def _normalize_models(
    models: str | Sequence[str] | Mapping[str, SummaryProvider],
    summary_generator: SummaryGenerator | None,
) -> dict[str, SummaryProvider | None]:
    if isinstance(models, Mapping):
        normalized = {str(name).strip(): source for name, source in models.items()}
    else:
        names = [models] if isinstance(models, str) else list(models)
        normalized = {str(name).strip(): None for name in names}
        if summary_generator is None:
            raise EvaluationError(
                "Model names require a summary_generator; alternatively pass a mapping of names to summaries."
            )
    if not normalized or any(not name for name in normalized):
        raise EvaluationError("At least one non-empty model name is required.")
    return normalized


def _normalize_packets(
    packets: Sequence[str | Mapping[str, Any] | EvidencePacket] | None,
) -> list[EvidencePacket]:
    normalized: list[EvidencePacket] = []
    for index, packet in enumerate(packets or [], start=1):
        if isinstance(packet, EvidencePacket):
            item = packet
        elif isinstance(packet, str):
            item = EvidencePacket(id=f"packet-{index}", text=packet.strip())
        elif isinstance(packet, Mapping):
            text = str(packet.get("text") or packet.get("excerpt") or "").strip()
            packet_id = str(packet.get("id") or f"packet-{index}").strip()
            item = EvidencePacket(id=packet_id, text=text)
        else:
            raise EvaluationError(f"Evidence packet {index} has an unsupported type.")
        if not item.id or not item.text:
            raise EvaluationError(f"Evidence packet {index} must have a non-empty id and text.")
        normalized.append(item)
    return normalized


def _summary_text(value: SummaryValue) -> str:
    if isinstance(value, Mapping):
        value = value.get("summary", "")
    text = str(value or "").strip()
    if not text:
        raise EvaluationError("Generated summaries must contain non-empty summary text.")
    return text


def _extract_json(raw_text: str) -> str:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group(0) if match else cleaned


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _missing_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_dimension(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    covered = _nonnegative_int(source.get("covered"))
    partial = _nonnegative_int(source.get("partial_covered"))
    missing = _missing_items(source.get("missing"))
    minimum_expected = covered + partial + len(missing)
    expected = max(_nonnegative_int(source.get("expected")), minimum_expected)
    return {
        "covered": covered,
        "partial_covered": partial,
        "expected": expected,
        "missing": missing,
    }


def _normalize_judgment(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError("The judge response must be a JSON object.")
    hallucinations: list[dict[str, Any]] = []
    raw_hallucinations = value.get("hallucinated_claims", [])
    if isinstance(raw_hallucinations, list):
        for item in raw_hallucinations:
            if not isinstance(item, Mapping) or not str(item.get("claim") or "").strip():
                continue
            hallucinations.append({
                "claim": str(item.get("claim")).strip(),
                "reason": str(item.get("reason") or "Not supported by evidence packet").strip(),
                "evidence": item.get("evidence"),
            })
    return {
        "key_ideas": _normalize_dimension(value.get("key_ideas")),
        "contributions": _normalize_dimension(value.get("contributions")),
        "hallucinated_claims": hallucinations,
    }


def _coverage_score(dimension: Mapping[str, Any]) -> float:
    expected = int(dimension["expected"])
    if expected == 0:
        return 1.0
    return min(1.0, (int(dimension["covered"]) + 0.5 * int(dimension["partial_covered"])) / expected)


def _score_result(judgment: Mapping[str, Any]) -> dict[str, float]:
    key_score = _coverage_score(judgment["key_ideas"])
    contribution_score = _coverage_score(judgment["contributions"])
    hallucinated = len(judgment["hallucinated_claims"])
    total_claims = int(judgment["key_ideas"]["covered"]) + int(judgment["contributions"]["covered"] + int(judgment["key_ideas"]["partial_covered"]) + int(judgment["contributions"]["partial_covered"]))
    hallucination_score = hallucinated / total_claims if total_claims else 0.0
    return {
        "avg_score": round(((key_score + contribution_score) / 2) - hallucination_score, 4),
        "key_ideas_score": round(key_score, 4),
        "contributions_score": round(contribution_score, 4),
        "hallucination_score": round(hallucination_score, 4),
    }


def _average_results(packet_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[Mapping[str, float]]] = {}
    for item in packet_results:
        by_model.setdefault(str(item["model"]), []).append(item["evaluation"])

    results: list[dict[str, Any]] = []
    for model_name, evaluations in by_model.items():
        averaged = {
            field: round(
                sum(float(evaluation[field]) for evaluation in evaluations) / len(evaluations),
                4,
            )
            for field in SCORE_FIELDS
        }
        results.append({
            "model": model_name,
            "packet_count": len(evaluations),
            "evaluation": averaged,
        })
    return results


def _group_packet_results_by_model(
    packet_results: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Shape packet-level output as ``{model_id: [evaluations...]}``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in packet_results:
        grouped.setdefault(str(item["model"]), []).append({
            "packet_id": str(item["packet_id"]),
            "evaluation": item["evaluation"],
        })
    return grouped


def save_average_scores_chart(
    average_results: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_EVALUATION_DIR,
) -> Path:
    """Save a PNG grouped bar chart comparing average model scores."""
    if not average_results:
        raise EvaluationError("Average-score visualization requires at least one model.")

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = Path(gettempdir()) / "deepdoc-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise EvaluationError(
            "Matplotlib is required to generate evaluation visualizations."
        ) from error

    labels = {
        "avg_score": "Overall",
        "key_ideas_score": "Key ideas",
        "contributions_score": "Contributions",
        "hallucination_score": "Hallucination",
    }
    colors = ("#2563eb", "#16a34a", "#9333ea", "#dc2626", "#0891b2", "#ca8a04")
    model_count = len(average_results)
    x_positions = list(range(len(SCORE_FIELDS)))
    bar_width = 0.8 / model_count
    figure, axis = plt.subplots(figsize=(max(8, model_count * 1.5), 5.5))
    try:
        for model_index, item in enumerate(average_results):
            offset = (model_index - (model_count - 1) / 2) * bar_width
            positions = [position + offset for position in x_positions]
            values = [
                max(0.0, min(1.0, float(item["evaluation"][field])))
                for field in SCORE_FIELDS
            ]
            axis.bar(
                positions,
                values,
                width=bar_width,
                label=str(item["model"]),
                color=colors[model_index % len(colors)],
            )

        axis.set_title("Average summary evaluation scores")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.set_xticks(x_positions, [labels[field] for field in SCORE_FIELDS])
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
        figure.text(
            0.99,
            0.01,
            "Higher is better except hallucination score",
            ha="right",
            fontsize=8,
            color="#4b5563",
        )
        figure.tight_layout(rect=(0, 0.08, 1, 1))

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        chart_path = target_dir / f"model-comparison-{timestamp}-{uuid4().hex[:8]}.png"
        figure.savefig(chart_path, dpi=160, bbox_inches="tight")
    except (OSError, ValueError) as error:
        raise EvaluationError(f"Could not save evaluation visualization: {error}") from error
    finally:
        plt.close(figure)

    return chart_path


def _detailed_result(judgment: Mapping[str, Any]) -> dict[str, Any]:
    key_ideas = judgment["key_ideas"]
    contributions = judgment["contributions"]
    return {
        "key_ideas": {
            "covered": key_ideas["covered"],
            "expected": key_ideas["expected"],
            "missing": key_ideas["missing"],
        },
        "contributions": {
            "covered": contributions["covered"],
            "missing": max(
                len(contributions["missing"]),
                contributions["expected"]
                - contributions["covered"]
                - contributions["partial_covered"],
            ),
        },
        "hallucinated_claims": judgment["hallucinated_claims"],
    }
