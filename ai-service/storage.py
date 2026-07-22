import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parent / "data" / "analyses"
UPLOAD_DIR = BASE_DIR / "uploads"
VIDEO_DIR = BASE_DIR / "data" / "videos"
VIDEO_WORK_DIR = VIDEO_DIR / "work"
ANALYSIS_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = (
    os.getenv("SUPABASE_SECRET_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or ""
)
SUPABASE_ANALYSES_TABLE = os.getenv("SUPABASE_ANALYSES_TABLE", "analyses")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def is_supabase_storage_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY)


def storage_backend_name() -> str:
    return "supabase" if is_supabase_storage_enabled() else "local_json"


def _supabase_headers(prefer: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _supabase_table_url(query: str = "") -> str:
    base = f"{SUPABASE_URL}/rest/v1/{urllib.parse.quote(SUPABASE_ANALYSES_TABLE)}"
    return f"{base}?{query}" if query else base


def _supabase_request(
    method: str,
    query: str = "",
    payload: Any | None = None,
    prefer: str | None = None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _supabase_table_url(query),
        data=data,
        headers=_supabase_headers(prefer=prefer),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supabase storage request failed: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Supabase storage request failed: {error.reason}") from error

    return json.loads(body) if body else None


def _summary_metadata(record: dict) -> dict:
    result = record.get("result", {})
    summary = result.get("document_summary", {}) if isinstance(result, dict) else {}
    source = result.get("source_metadata", {}) if isinstance(result.get("source_metadata"), dict) else {}
    return {
        "paper_title": source.get("title") or result.get("paper_title") or summary.get("title"),
        "summary_mode": result.get("summary_mode"),
        "processing_seconds": result.get("processing_seconds"),
        "summary": summary.get("summary", ""),
    }


def _supabase_row_from_record(record: dict, user_id: str) -> dict:
    metadata = _summary_metadata(record)
    return {
        "analysis_id": record.get("analysis_id"),
        "user_id": user_id,
        "filename": record.get("filename"),
        "paper_title": metadata.get("paper_title"),
        "summary_mode": metadata.get("summary_mode"),
        "processing_seconds": metadata.get("processing_seconds"),
        "summary": metadata.get("summary"),
        "created_at": record.get("created_at"),
        "record": record,
    }


def _record_belongs_to_user(record: dict, user_id: str | None) -> bool:
    return not user_id or record.get("user_id") == user_id


def save_analysis(
    filename: str,
    result: dict,
    source_pdf_path: str | Path | None = None,
    user_id: str | None = None,
) -> dict:
    _ensure_data_dir()

    analysis_id = uuid4().hex
    result["analysis_id"] = analysis_id
    record = {
        "analysis_id": analysis_id,
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    if user_id:
        record["user_id"] = user_id
    if source_pdf_path:
        record["source_pdf_path"] = str(source_pdf_path)

    if is_supabase_storage_enabled():
        if not user_id:
            raise RuntimeError("Supabase storage requires a user_id.")
        rows = _supabase_request(
            "POST",
            payload=_supabase_row_from_record(record, user_id),
            prefer="return=representation",
        )
        return rows[0].get("record", record) if isinstance(rows, list) and rows else record

    record_path = DATA_DIR / f"{analysis_id}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return record


def list_analyses(user_id: str | None = None) -> list[dict]:
    _ensure_data_dir()
    analyses: list[dict] = []

    if is_supabase_storage_enabled():
        if not user_id:
            return []
        query = urllib.parse.urlencode({
            "user_id": f"eq.{user_id}",
            "select": "analysis_id,filename,paper_title,created_at,summary_mode,processing_seconds,summary",
            "order": "created_at.desc",
        })
        rows = _supabase_request("GET", query=query)
        return rows if isinstance(rows, list) else []

    for record_path in DATA_DIR.glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _record_belongs_to_user(record, user_id):
            continue

        result = record.get("result", {})
        summary = result.get("document_summary", {})
        source = result.get("source_metadata", {}) if isinstance(result.get("source_metadata"), dict) else {}
        analyses.append({
            "analysis_id": record.get("analysis_id"),
            "filename": record.get("filename"),
            "paper_title": source.get("title") or result.get("paper_title") or summary.get("title"),
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


def get_analysis(analysis_id: str, user_id: str | None = None) -> Optional[dict]:
    _ensure_data_dir()
    if not ANALYSIS_ID_RE.match(analysis_id):
        return None

    if is_supabase_storage_enabled():
        if not user_id:
            return None
        query = urllib.parse.urlencode({
            "analysis_id": f"eq.{analysis_id}",
            "user_id": f"eq.{user_id}",
            "select": "record",
            "limit": "1",
        })
        rows = _supabase_request("GET", query=query)
        if not isinstance(rows, list) or not rows:
            return None
        record = rows[0].get("record")
        return record if isinstance(record, dict) else None

    record_path = DATA_DIR / f"{analysis_id}.json"
    if not record_path.exists():
        return None

    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return record if _record_belongs_to_user(record, user_id) else None


def save_analysis_record(analysis_id: str, record: dict, user_id: str | None = None) -> Optional[dict]:
    _ensure_data_dir()
    if not ANALYSIS_ID_RE.match(analysis_id):
        return None
    if user_id and record.get("user_id") not in {None, user_id}:
        return None

    record["analysis_id"] = analysis_id
    if user_id:
        record["user_id"] = user_id
    result = record.get("result")
    if isinstance(result, dict):
        result["analysis_id"] = result.get("analysis_id") or analysis_id

    if is_supabase_storage_enabled():
        if not user_id:
            return None
        query = urllib.parse.urlencode({
            "analysis_id": f"eq.{analysis_id}",
            "user_id": f"eq.{user_id}",
        })
        rows = _supabase_request(
            "PATCH",
            query=query,
            payload=_supabase_row_from_record(record, user_id),
            prefer="return=representation",
        )
        if not isinstance(rows, list) or not rows:
            return None
        saved_record = rows[0].get("record")
        return saved_record if isinstance(saved_record, dict) else record

    record_path = DATA_DIR / f"{analysis_id}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _path_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _safe_unlink(path_value: object, allowed_dirs: list[Path]) -> Optional[str]:
    if not path_value:
        return None

    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return None
    if not any(_path_inside(path, allowed_dir) for allowed_dir in allowed_dirs):
        return None

    path.unlink()
    return str(path)


def _safe_rmtree(path: Path, allowed_parent: Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    if not _path_inside(path, allowed_parent):
        return None

    shutil.rmtree(path)
    return str(path)


def _delete_associated_files(record: dict) -> list[str]:
    deleted_paths: list[str] = []
    source_pdf = _safe_unlink(record.get("source_pdf_path"), [UPLOAD_DIR])
    if source_pdf:
        deleted_paths.append(source_pdf)
    elif not record.get("source_pdf_path"):
        legacy_pdf = _legacy_pdf_path(record)
        if legacy_pdf and not _legacy_pdf_is_shared(record, legacy_pdf):
            deleted_legacy_pdf = _safe_unlink(legacy_pdf, [UPLOAD_DIR])
            if deleted_legacy_pdf:
                deleted_paths.append(deleted_legacy_pdf)

    result = record.get("result", {})
    video = result.get("video", {}) if isinstance(result, dict) else {}
    video_path = video.get("video_path") if isinstance(video, dict) else None
    deleted_video = _safe_unlink(video_path, [VIDEO_DIR])
    if deleted_video:
        deleted_paths.append(deleted_video)

    analysis_id = str(record.get("analysis_id") or result.get("analysis_id") or "")
    if ANALYSIS_ID_RE.match(analysis_id):
        work_dir = _safe_rmtree(VIDEO_WORK_DIR / analysis_id, VIDEO_WORK_DIR)
        if work_dir:
            deleted_paths.append(work_dir)

    return deleted_paths


def _legacy_pdf_path(record: dict) -> Optional[Path]:
    filename = Path(record.get("filename") or "").name
    if not filename:
        return None
    path = UPLOAD_DIR / filename
    return path if path.exists() else None


def _legacy_pdf_is_shared(record: dict, pdf_path: Path) -> bool:
    current_id = str(record.get("analysis_id") or "")
    current_filename = Path(record.get("filename") or "").name
    for record_path in DATA_DIR.glob("*.json"):
        try:
            other_record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(other_record.get("analysis_id") or "") == current_id:
            continue
        other_source = other_record.get("source_pdf_path")
        if other_source and Path(str(other_source)).resolve(strict=False) == pdf_path.resolve(strict=False):
            return True
        if not other_source and Path(other_record.get("filename") or "").name == current_filename:
            return True
    return False


def delete_analysis(analysis_id: str, delete_files: bool = True, user_id: str | None = None) -> Optional[dict]:
    _ensure_data_dir()
    if not ANALYSIS_ID_RE.match(analysis_id):
        return None

    if is_supabase_storage_enabled():
        if not user_id:
            return None
        record = get_analysis(analysis_id, user_id=user_id)
        if not record:
            return None
        deleted_files = _delete_associated_files(record) if delete_files else []
        query = urllib.parse.urlencode({
            "analysis_id": f"eq.{analysis_id}",
            "user_id": f"eq.{user_id}",
        })
        _supabase_request("DELETE", query=query)
        return {
            "analysis_id": analysis_id,
            "deleted": True,
            "deleted_files": deleted_files,
        }

    record_path = DATA_DIR / f"{analysis_id}.json"
    if not record_path.exists():
        return None

    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        record = {"analysis_id": analysis_id}
    if not _record_belongs_to_user(record, user_id):
        return None

    deleted_files = _delete_associated_files(record) if delete_files else []

    record_path.unlink()
    return {
        "analysis_id": analysis_id,
        "deleted": True,
        "deleted_files": deleted_files,
    }


def save_video_script(analysis_id: str, video_script: dict, user_id: str | None = None) -> Optional[dict]:
    record = get_analysis(analysis_id, user_id=user_id)
    if not record:
        return None

    result = record.setdefault("result", {})
    result["video_script"] = video_script
    result["video_script_generated_at"] = datetime.now(timezone.utc).isoformat()
    result.pop("video", None)

    return save_analysis_record(analysis_id, record, user_id=user_id)

def save_video_result(analysis_id: str, video_result: dict, user_id: str | None = None) -> Optional[dict]:
    record = get_analysis(analysis_id, user_id=user_id)
    if not record:
        return None

    result = record.setdefault("result", {})
    result["video"] = video_result

    return save_analysis_record(analysis_id, record, user_id=user_id)
