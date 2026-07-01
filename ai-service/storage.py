import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4


DATA_DIR = Path(__file__).resolve().parent / "data" / "analyses"
ANALYSIS_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_analysis(filename: str, result: dict) -> dict:
    _ensure_data_dir()

    analysis_id = uuid4().hex
    result["analysis_id"] = analysis_id
    record = {
        "analysis_id": analysis_id,
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }

    record_path = DATA_DIR / f"{analysis_id}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return record


def update_analysis_result(analysis_id: str, result: dict) -> Optional[dict]:
    record = get_analysis(analysis_id)
    if not record:
        return None

    result["analysis_id"] = analysis_id
    record["result"] = result
    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    record_path = DATA_DIR / f"{analysis_id}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def list_analyses() -> list[dict]:
    _ensure_data_dir()
    analyses: list[dict] = []

    for record_path in DATA_DIR.glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        result = record.get("result", {})
        summary = result.get("document_summary", {})
        analyses.append({
            "analysis_id": record.get("analysis_id"),
            "filename": record.get("filename"),
            "paper_title": result.get("paper_title") or summary.get("title"),
            "created_at": record.get("created_at"),
            "summary_mode": result.get("summary_mode"),
            "processing_seconds": result.get("processing_seconds"),
            "summary": summary.get("summary", ""),
        })

    return sorted(
        analyses,
        key=lambda analysis: analysis.get("created_at") or "",
        reverse=True,
    )


def get_analysis(analysis_id: str) -> Optional[dict]:
    _ensure_data_dir()
    if not ANALYSIS_ID_RE.match(analysis_id):
        return None

    record_path = DATA_DIR / f"{analysis_id}.json"
    if not record_path.exists():
        return None

    try:
        return json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def delete_analysis(analysis_id: str) -> bool:
    _ensure_data_dir()
    if not ANALYSIS_ID_RE.match(analysis_id):
        return False

    record_path = DATA_DIR / f"{analysis_id}.json"
    if not record_path.exists():
        return False

    record_path.unlink()
    return True


def save_video_script(analysis_id: str, video_script: dict) -> Optional[dict]:
    record = get_analysis(analysis_id)
    if not record:
        return None

    result = record.setdefault("result", {})
    result["video_script"] = video_script
    result["video_script_generated_at"] = datetime.now(timezone.utc).isoformat()
    result.pop("video", None)

    record_path = DATA_DIR / f"{analysis_id}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record

def save_video_result(analysis_id: str, video_result: dict) -> Optional[dict]:
    record = get_analysis(analysis_id)
    if not record:
        return None

    result = record.setdefault("result", {})
    result["video"] = video_result

    record_path = DATA_DIR / f"{analysis_id}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
