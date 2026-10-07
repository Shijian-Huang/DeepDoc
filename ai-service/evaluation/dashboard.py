"""Lazy evaluation orchestration and HTTP routes for the evaluator dashboard."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evaluation.models import EvaluationModel, model_registry
from llm.evaluator import (
    DEFAULT_PACKETS_PATH,
    EvidencePacket,
    EvaluationJudge,
    GeminiJudge,
    SCORE_FIELDS,
    SummaryEvaluator,
    load_default_evidence_packets,
)
from parser.document_parser import SUPPORTED_EXTENSIONS, parse_document_pages
from utils.section_extractor import build_summary_input_from_pages
from utils.summary_prompt import SUMMARY_MODE_INSTRUCTIONS, normalize_summary_mode


BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
DEFAULT_CACHE_PATH = BASE_DIR / "data" / "eval" / "dashboard_results.json"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


class DashboardRequest(BaseModel):
    model_names: list[str] = Field(min_length=1)
    paper_ids: list[str] | None = None
    summary_mode: str = "standard"
    force: bool = False


class JsonResultStore:
    """Small process-safe cache persisted atomically to local disk."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._read().get(key)
            return dict(value) if isinstance(value, Mapping) else None

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        with self._lock:
            payload = self._read()
            payload[key] = dict(value)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}


@dataclass
class DashboardEvaluator:
    """Evaluate only requested model/paper pairs and reuse prior results."""

    models: Mapping[str, EvaluationModel]
    packets: Sequence[EvidencePacket]
    judge: EvaluationJudge
    store: JsonResultStore

    @classmethod
    def configured(cls) -> "DashboardEvaluator":
        return cls(
            models=model_registry(),
            packets=load_default_evidence_packets(DEFAULT_PACKETS_PATH),
            judge=GeminiJudge(),
            store=JsonResultStore(),
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "papers": [{"id": packet.id, "label": _paper_label(packet.id)} for packet in self.packets],
            "summary_modes": list(SUMMARY_MODE_INSTRUCTIONS),
            "score_fields": list(SCORE_FIELDS),
        }

    def evaluate(
        self,
        model_names: Sequence[str],
        paper_ids: Sequence[str] | None,
        summary_mode: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        selected_models = self._select_models(model_names)
        selected_packets = self._select_packets(paper_ids)
        normalized_mode = _validate_summary_mode(summary_mode)
        rows = self._evaluate_pairs(
            selected_models,
            selected_packets,
            normalized_mode,
            force=force,
        )
        return _response(rows, normalized_mode, averaged=len(selected_packets) > 1)

    def evaluate_custom(
        self,
        summary: str,
        packet: EvidencePacket,
        summary_mode: str,
        model_names: Sequence[str] = (),
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = _validate_summary_mode(summary_mode)
        summary_text = summary.strip()
        if not summary_text:
            raise ValueError("A custom summary is required.")

        custom_model = _StaticSummaryModel("Custom summary", summary_text)
        selected_models = [
            custom_model,
            *self._select_models(model_names, allow_empty=True),
        ]
        rows = self._evaluate_pairs(
            selected_models,
            [packet],
            normalized_mode,
            force=force,
        )
        return _response(rows, normalized_mode, averaged=False)

    def _evaluate_pairs(
        self,
        models: Sequence[EvaluationModel],
        packets: Sequence[EvidencePacket],
        summary_mode: str,
        *,
        force: bool,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        isolate_errors = len(models) * len(packets) > 1
        for model in models:
            for packet in packets:
                try:
                    rows.append(
                        self._evaluate_pair(
                            model,
                            packet,
                            summary_mode,
                            force=force,
                        )
                    )
                except Exception as error:
                    if not isolate_errors:
                        raise
                    rows.append({
                        "model": model.name,
                        "paper_id": packet.id,
                        "error": _evaluation_error_message(error),
                        "cached": False,
                        "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    })
        return rows

    def _evaluate_pair(
        self,
        model: EvaluationModel,
        packet: EvidencePacket,
        summary_mode: str,
        *,
        force: bool,
    ) -> dict[str, Any]:
        key = _cache_key(model.name, packet, summary_mode, getattr(model, "cache_identity", ""))
        if not force:
            cached = self.store.get(key)
            if cached is not None:
                cached["cached"] = True
                return cached

        result = SummaryEvaluator(
            {model.name: model.summarize},
            [packet],
            judge=self.judge,
            summary_mode=summary_mode,
        ).evaluate()[model.name][0]
        row = {
            "model": model.name,
            "paper_id": packet.id,
            "evaluation": result["evaluation"],
            "cached": False,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.store.put(key, row)
        return row

    def _select_models(
        self,
        names: Sequence[str],
        *,
        allow_empty: bool = False,
    ) -> list[EvaluationModel]:
        deduplicated = list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))
        unknown = [name for name in deduplicated if name not in self.models]
        if unknown:
            raise ValueError(f"Unknown evaluation model(s): {', '.join(unknown)}")
        if not deduplicated and not allow_empty:
            raise ValueError("Select at least one model.")
        return [self.models[name] for name in deduplicated]

    def _select_packets(self, ids: Sequence[str] | None) -> list[EvidencePacket]:
        packet_by_id = {packet.id: packet for packet in self.packets}
        selected_ids = list(dict.fromkeys(ids or packet_by_id))
        unknown = [packet_id for packet_id in selected_ids if packet_id not in packet_by_id]
        if unknown:
            raise ValueError(f"Unknown paper(s): {', '.join(unknown)}")
        if not selected_ids:
            raise ValueError("Select at least one paper.")
        return [packet_by_id[packet_id] for packet_id in selected_ids]


@dataclass(frozen=True)
class _StaticSummaryModel:
    name: str
    value: str

    @property
    def cache_identity(self) -> str:
        return hashlib.sha256(self.value.encode("utf-8")).hexdigest()

    def summarize(self, _prompt: str) -> str:
        return self.value


def _validate_summary_mode(value: str) -> str:
    if value not in SUMMARY_MODE_INSTRUCTIONS:
        raise ValueError(f"Unknown summary mode: {value}")
    return normalize_summary_mode(value)


def _cache_key(model_name: str, packet: EvidencePacket, summary_mode: str, identity: str) -> str:
    value = json.dumps(
        [model_name, summary_mode, packet.id, packet.text, identity],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _averages(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, float]]] = {}
    errors: dict[str, list[str]] = {}
    for row in rows:
        model = str(row["model"])
        if "evaluation" in row:
            grouped.setdefault(model, []).append(row["evaluation"])
        elif row.get("error"):
            errors.setdefault(model, []).append(str(row["error"]))

    results: list[dict[str, Any]] = []
    for model in dict.fromkeys(str(row["model"]) for row in rows):
        evaluations = grouped.get(model, [])
        model_errors = errors.get(model, [])
        if not evaluations:
            results.append({
                "model": model,
                "paper_count": 0,
                "error": model_errors[0] if model_errors else "No evaluation was produced.",
                "failed_paper_count": len(model_errors),
            })
            continue
        result: dict[str, Any] = {
            "model": model,
            "paper_count": len(evaluations),
            "evaluation": {
                field: round(sum(float(item[field]) for item in evaluations) / len(evaluations), 4)
                for field in SCORE_FIELDS
            },
        }
        if model_errors:
            result["failed_paper_count"] = len(model_errors)
            result["warning"] = f"{len(model_errors)} paper evaluation(s) failed."
        results.append(result)
    return results


def _response(rows: list[dict[str, Any]], summary_mode: str, *, averaged: bool) -> dict[str, Any]:
    failed_rows = [row for row in rows if row.get("error")]
    return {
        "summary_mode": summary_mode,
        "averaged": averaged,
        "results": _averages(rows) if averaged else rows,
        "packet_results": rows,
        "partial": bool(failed_rows),
        "error_count": len(failed_rows),
    }


def _evaluation_error_message(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        raw_response = str(error.doc or "").strip()
        if not raw_response:
            return "The model returned an empty response instead of a JSON summary."
        excerpt = " ".join(raw_response.split())[:240]
        return f"The model returned an invalid JSON summary: {excerpt}"
    return str(error).strip() or error.__class__.__name__


def _paper_label(paper_id: str) -> str:
    return Path(paper_id).stem.replace("_", " ").replace("-", " ").title()


def build_custom_packet(path: str | Path, summary_mode: str, filename: str = "upload") -> EvidencePacket:
    pages = parse_document_pages(str(path))
    if not pages:
        raise ValueError("The uploaded reference paper contains no readable text.")
    packet_text, _sections, _sources = build_summary_input_from_pages(
        pages,
        summary_mode=_validate_summary_mode(summary_mode),
    )
    if not packet_text.strip():
        raise ValueError("Could not build an evidence packet from the uploaded paper.")
    digest = hashlib.sha256(packet_text.encode("utf-8")).hexdigest()[:12]
    return EvidencePacket(id=f"custom:{Path(filename).name}:{digest}", text=packet_text)


router = APIRouter()
dashboard_evaluator = DashboardEvaluator.configured()


@router.get("/evaluator-dashboard", include_in_schema=False)
async def evaluator_dashboard_page():
    return FileResponse(
        STATIC_DIR / "evaluator.html",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/api/evaluator/config")
async def evaluator_configuration():
    return dashboard_evaluator.configuration()


@router.post("/api/evaluator/evaluate")
async def evaluate_selection(request: DashboardRequest):
    try:
        return dashboard_evaluator.evaluate(
            request.model_names,
            request.paper_ids,
            request.summary_mode,
            force=request.force,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {error}") from error


@router.post("/api/evaluator/custom")
async def evaluate_custom_summary(
    paper: UploadFile = File(...),
    summary: str = Form(...),
    summary_mode: str = Form("standard"),
    model_names: str = Form("[]"),
    force: bool = Form(False),
):
    filename = Path(paper.filename or "reference.pdf").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported reference-paper format.")
    try:
        selected_models = json.loads(model_names)
        if not isinstance(selected_models, list):
            raise ValueError("model_names must be a JSON array.")
        content = await paper.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("The reference paper exceeds the 30 MB upload limit.")
        if not content:
            raise ValueError("The uploaded reference paper is empty.")
        with NamedTemporaryFile(suffix=suffix) as temporary:
            temporary.write(content)
            temporary.flush()
            packet = build_custom_packet(temporary.name, summary_mode, filename)
        return dashboard_evaluator.evaluate_custom(
            summary,
            packet,
            summary_mode,
            selected_models,
            force=force,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {error}") from error
