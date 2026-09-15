from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

DEFAULT_PACKETS_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "default_packets.json"
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
SummaryProvider = SummaryValue | Callable[[EvidencePacket], SummaryValue]
SummaryGenerator = Callable[[str, EvidencePacket], SummaryValue]


class GeminiJudge:
    """Gemini-backed judge; replace this class through dependency injection."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_EVALUATOR_MODEL", "gemini-2.5-flash-lite")
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
    summary or a callable that generates a summary for a packet.  A model name
    (or sequence of names) can instead be supplied with ``summary_generator``.
    """

    def __init__(
        self,
        models: str | Sequence[str] | Mapping[str, SummaryProvider],
        evidence_packets: Sequence[str | Mapping[str, Any] | EvidencePacket] | None = None,
        *,
        judge: EvaluationJudge | None = None,
        summary_generator: SummaryGenerator | None = None,
        default_packets_path: str | Path | None = None,
    ) -> None:
        self._model_sources = _normalize_models(models, summary_generator)
        self._judge = judge or GeminiJudge()
        self._summary_generator = summary_generator
        self._default_packets_path = Path(default_packets_path or DEFAULT_PACKETS_PATH)
        self._packets = (
            _normalize_packets(evidence_packets)
            if evidence_packets is not None
            else load_default_evidence_packets(self._default_packets_path)
        )
        if not self._packets:
            raise EvaluationError("At least one evidence packet is required.")

    def evaluate(self, mode: str = "default") -> dict[str, Any]:
        """Return a JSON-compatible result for all model/packet combinations."""
        if mode not in {"default", "detailed"}:
            raise EvaluationError("mode must be either 'default' or 'detailed'.")

        results: list[dict[str, Any]] = []
        for model_name, source in self._model_sources.items():
            for packet in self._packets:
                summary = self._resolve_summary(model_name, source, packet)
                judgment = _normalize_judgment(self._judge.evaluate(summary, packet))
                evaluation = (
                    _detailed_result(judgment)
                    if mode == "detailed"
                    else _score_result(judgment)
                )
                results.append({
                    "model": model_name,
                    "packet_id": packet.id,
                    "evaluation": evaluation,
                })

        return {"mode": mode, "results": results}

    def evaluate_json(self, mode: str = "default", *, indent: int | None = None) -> str:
        """Return the evaluation as serialized, standards-compliant JSON."""
        return json.dumps(self.evaluate(mode), ensure_ascii=False, indent=indent, allow_nan=False)

    def _resolve_summary(
        self,
        model_name: str,
        source: SummaryProvider | None,
        packet: EvidencePacket,
    ) -> str:
        if callable(source):
            value = source(packet)
        elif source is not None:
            value = source
        elif self._summary_generator is not None:
            value = self._summary_generator(model_name, packet)
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
    expected = int(judgment["key_ideas"]["expected"]) + int(
        judgment["contributions"]["expected"]
    )
    hallucination_score = hallucinated / (expected + hallucinated) if expected + hallucinated else 0.0
    return {
        "avg_score": round((key_score + contribution_score) / 2, 4),
        "key_ideas_score": round(key_score, 4),
        "contributions_score": round(contribution_score, 4),
        "hallucination_score": round(hallucination_score, 4),
    }


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
