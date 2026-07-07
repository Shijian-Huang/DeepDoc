from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import html
import json
import re
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

from llm.summarizer import generate_video_script, gemini_configuration_error, is_gemini_configured
from pipeline import run_pipeline
from services.arxiv_service import (
    ArxivServiceError,
    arxiv_id_from_pdf_url,
    download_arxiv_pdf,
    is_valid_arxiv_pdf_url,
    search_arxiv,
)
from storage import delete_analysis, get_analysis, list_analyses, save_analysis, save_video_result, save_video_script
from video_generator import (
    PIPER_BIN,
    PIPER_MODEL,
    TTS_PROVIDER,
    VideoGenerationError,
    generate_video_from_script,
)

app = FastAPI(
    title="DeepDoc",
    description="AI-powered research paper analysis service.",
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
DOWNLOAD_TIMEZONE = ZoneInfo("America/Los_Angeles")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ArxivAnalyzeRequest(BaseModel):
    arxiv_id: str
    pdf_url: str
    summary_mode: str = "standard"


def analyze_pdf_file(file_path: str | Path, filename: str, summary_mode: str = "standard") -> dict:
    _ensure_gemini_configured()

    started_at = time.perf_counter()
    submitted_at = datetime.now(timezone.utc)

    try:
        result = run_pipeline(str(file_path), summary_mode=summary_mode)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}") from error

    generated_at = datetime.now(timezone.utc)
    result["submitted_at"] = submitted_at.isoformat()
    result["generated_at"] = generated_at.isoformat()
    result["processing_seconds"] = round(time.perf_counter() - started_at, 2)
    record = save_analysis(filename, result, source_pdf_path=Path(file_path))
    return record["result"]


def _ensure_gemini_configured() -> None:
    if not is_gemini_configured():
        raise HTTPException(status_code=503, detail=gemini_configuration_error())


def _remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass


def _tts_health() -> dict:
    provider = TTS_PROVIDER
    if provider == "piper":
        venv_piper = Path(sys.executable).parent / "piper"
        piper_binary_ready = shutil.which(PIPER_BIN) is not None or venv_piper.exists()
        model_ready = Path(PIPER_MODEL).expanduser().exists()
        return {
            "provider": provider,
            "ready": piper_binary_ready and model_ready,
            "piper_binary_available": piper_binary_ready,
            "piper_model_available": model_ready,
        }
    if provider == "say":
        say_ready = shutil.which("say") is not None
        return {
            "provider": provider,
            "ready": say_ready,
            "say_available": say_ready,
        }
    if provider == "openai":
        import os
        openai_ready = bool(os.getenv("OPENAI_API_KEY"))
        return {
            "provider": provider,
            "ready": openai_ready,
            "openai_api_key_configured": openai_ready,
        }
    return {
        "provider": provider,
        "ready": False,
        "error": "Unknown TTS provider.",
    }


def _download_timestamp(record: dict) -> str:
    result = record.get("result", {})
    timestamp = (
        result.get("generated_at")
        or result.get("submitted_at")
        or record.get("created_at")
        or ""
    )
    timestamp_text = str(timestamp)
    try:
        parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError:
        return _format_download_timestamp(datetime.now(timezone.utc))

    return _format_download_timestamp(parsed)


def _format_download_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    local_time = value.astimezone(DOWNLOAD_TIMEZONE)
    return local_time.strftime("%Y%m%d-%H%M%S-%Z")


def _record_stem(record: dict) -> str:
    return _safe_filename_part(Path(record.get("filename") or "analysis").stem)


def _safe_filename_part(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip(".-")
    return text or "analysis"


def _safe_pdf_storage_name(filename: str) -> str:
    original = Path(filename or "uploaded.pdf")
    stem = _safe_filename_part(original.stem or "paper")
    suffix = original.suffix.lower() if original.suffix.lower() == ".pdf" else ".pdf"
    return f"{stem}-{uuid4().hex[:12]}{suffix}"


def _normalized_arxiv_id(value: str) -> str:
    return re.sub(r"v\d+$", "", str(value or "").strip().lower())


def _arxiv_version(value: str) -> str:
    match = re.search(r"(v\d+)$", str(value or "").strip().lower())
    return match.group(1) if match else ""


def _arxiv_ids_match(left: str, right: str) -> bool:
    left_text = str(left or "").strip().lower()
    right_text = str(right or "").strip().lower()
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if _normalized_arxiv_id(left_text) != _normalized_arxiv_id(right_text):
        return False
    return not _arxiv_version(left_text) or not _arxiv_version(right_text)


def _summary_mode_slug(record: dict) -> str:
    result = record.get("result", {})
    mode = result.get("summary_mode") or record.get("summary_mode") or "standard"
    return {
        "paragraph": "paragraph",
        "standard": "standard",
        "one_page": "onepage",
    }.get(mode, "standard")


def _artifact_timestamp(record: dict, artifact_key: str | None = None) -> str:
    result = record.get("result", {})
    timestamp = ""
    if artifact_key in {"video_script", "video-script", "slides", "slides-html"}:
        timestamp = result.get("video_script_generated_at") or ""
    elif artifact_key == "video":
        video = result.get("video", {})
        if isinstance(video, dict):
            timestamp = video.get("generated_at") or ""

    if not timestamp:
        return _download_timestamp(record)

    timestamp_text = str(timestamp)
    try:
        parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError:
        return _download_timestamp(record)

    return _format_download_timestamp(parsed)


def _artifact_filename(record: dict, analysis_id: str, artifact: str, extension: str) -> str:
    stem = _record_stem(record)
    timestamp = _artifact_timestamp(record, artifact)
    short_id = analysis_id[:8]
    result = record.get("result", {})
    mode = _summary_mode_slug(record)
    label = {
        "analysis": f"analysis-{extension}",
        "video-script": "slides-json",
        "slides": "slides-md",
        "slides-html": "slides-html",
        "video": "video-mp4",
        "source-pdf": "source-pdf",
    }.get(artifact, artifact)
    variant_parts: list[str] = []
    if artifact != "source-pdf":
        variant_parts.append(mode)
    if artifact in {"video-script", "slides", "slides-html"}:
        script = result.get("video_script", {})
        if isinstance(script, dict):
            slide_count = script.get("slide_count") or len(script.get("scenes") or [])
            if slide_count:
                variant_parts.append(f"{slide_count}slides")
    elif artifact == "video":
        script = result.get("video_script", {})
        video = result.get("video", {})
        slide_count = None
        if isinstance(script, dict):
            slide_count = script.get("slide_count") or len(script.get("scenes") or [])
        if not slide_count and isinstance(video, dict):
            slide_count = video.get("scene_count")
        if slide_count:
            variant_parts.append(f"{slide_count}slides")
    variant = "-".join(_safe_filename_part(part) for part in variant_parts if part)
    parts = [stem, label]
    if variant:
        parts.append(variant)
    parts.extend([timestamp, short_id])
    return f"{'--'.join(parts)}.{extension}"


def _markdown_text(value: object) -> str:
    text = str(value or "").strip()
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _markdown_list(items: object, empty: str = "None.") -> str:
    if not isinstance(items, list) or not items:
        return empty

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("claim")
                or item.get("title")
                or item.get("summary")
                or item.get("excerpt")
                or ""
            )
        else:
            text = str(item or "")
        text = _markdown_text(text)
        if text:
            lines.append(f"{index}. {text}")
    return "\n".join(lines) if lines else empty


def _format_pages(value: object) -> str:
    if isinstance(value, list):
        pages = [str(page) for page in value if str(page).strip()]
        return ", ".join(pages) if pages else "unknown"
    text = str(value or "").strip()
    return text or "unknown"


def _markdown_evidence(items: object, empty: str = "None.") -> str:
    if not isinstance(items, list) or not items:
        return empty

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        claim = _markdown_text(
            item.get("claim")
            or item.get("summary")
            or item.get("title")
            or item.get("excerpt")
            or item.get("snippet")
            or item.get("quote")
        )
        snippet = _markdown_text(item.get("excerpt") or item.get("snippet") or item.get("quote"))
        section = _markdown_text(item.get("section") or item.get("section_title") or item.get("heading") or "unknown")
        pages = _format_pages(item.get("pages") or item.get("page_numbers") or item.get("page_number") or item.get("page"))
        if not claim and not snippet:
            continue
        lines.append(f"{index}. **{claim or 'Evidence'}**")
        lines.append(f"   - Section: {section}")
        lines.append(f"   - Pages: {pages}")
        if snippet and snippet != claim:
            lines.append(f"   - Snippet: {snippet}")
    return "\n".join(lines) if lines else empty


def _markdown_references(references: object) -> str:
    if not isinstance(references, list) or not references:
        return "None extracted."
    lines = []
    for index, reference in enumerate(references, start=1):
        text = _markdown_text(reference)
        text = re.sub(r"^\s*(?:\[\d+\]|\d{1,3}[.)])\s*", "", text)
        if text:
            lines.append(f"{index}. {text}")
    return "\n".join(lines) if lines else "None extracted."


def _analysis_markdown(record: dict, analysis_id: str) -> str:
    result = record.get("result", {})
    summary = result.get("document_summary", {}) if isinstance(result.get("document_summary"), dict) else {}
    title = (
        result.get("paper_title")
        or summary.get("title")
        or Path(record.get("filename") or "Analysis").stem
    )

    lines = [
        f"# {_markdown_text(title)}",
        "",
        "## Metadata",
        "",
        f"- File: {_markdown_text(record.get('filename'))}",
        f"- Analysis ID: {analysis_id}",
        f"- Summary mode: {_summary_mode_slug(record)}",
        f"- Generated: {_markdown_text(result.get('generated_at') or record.get('created_at'))}",
        f"- Processing seconds: {_markdown_text(result.get('processing_seconds'))}",
        "",
        "## Summary",
        "",
        _markdown_text(summary.get("summary") or "No summary returned."),
        "",
        "## Key Ideas",
        "",
        _markdown_list(summary.get("key_ideas")),
        "",
        "## Contributions",
        "",
        _markdown_list(summary.get("contributions")),
    ]

    optional_sections = [
        ("Limitations", summary.get("limitations")),
        ("Discussion Questions", summary.get("discussion_questions") or summary.get("discussionQuestions")),
        ("Reviewer Questions", summary.get("reviewer_questions") or summary.get("reviewerQuestions")),
    ]
    for heading, items in optional_sections:
        if isinstance(items, list) and items:
            lines.extend(["", f"## {heading}", "", _markdown_list(items)])

    lines.extend([
        "",
        "## Evidence",
        "",
        _markdown_evidence(summary.get("evidence")),
        "",
        "## Source Sections",
        "",
        _markdown_evidence(result.get("evidence_sources")),
        "",
        "## References",
        "",
        _markdown_references(result.get("references")),
    ])

    script = result.get("video_script")
    if isinstance(script, dict) and script.get("scenes"):
        lines.extend([
            "",
            "## Video Script Outline",
            "",
            _slides_markdown(record, analysis_id, include_frontmatter=False),
        ])

    return "\n".join(lines).strip() + "\n"


def _slides_markdown(record: dict, analysis_id: str, include_frontmatter: bool = True) -> str:
    result = record.get("result", {})
    script = result.get("video_script")
    if not isinstance(script, dict) or not isinstance(script.get("scenes"), list) or not script["scenes"]:
        raise HTTPException(status_code=404, detail="Video script not found. Generate a script before downloading slides.")

    title = _markdown_text(script.get("title") or result.get("paper_title") or Path(record.get("filename") or "Slides").stem)
    lines: list[str] = []
    if include_frontmatter:
        lines.extend([
            "---",
            "marp: true",
            "theme: default",
            "paginate: true",
            "size: 16:9",
            "---",
            "",
        ])

    lines.extend([
        f"# {title}",
        "",
        f"{len(script['scenes'])} slides",
    ])

    for scene in script["scenes"]:
        if not isinstance(scene, dict):
            continue
        lines.extend(["", "---", ""])
        heading = _markdown_text(scene.get("heading") or f"Slide {scene.get('scene_number', '')}".strip())
        lines.append(f"# {heading}")
        lines.append("")
        bullets = scene.get("bullets")
        if isinstance(bullets, list) and bullets:
            for bullet in bullets:
                text = _markdown_text(bullet)
                if text:
                    lines.append(f"- {text}")
        voiceover = _markdown_text(scene.get("voiceover"))
        if voiceover:
            lines.extend(["", "## Speaker Notes", "", voiceover])
        evidence = scene.get("evidence")
        if isinstance(evidence, dict):
            claim = _markdown_text(evidence.get("claim") or evidence.get("excerpt") or evidence.get("summary"))
            pages = _format_pages(evidence.get("pages") or evidence.get("page_numbers") or evidence.get("page_number") or evidence.get("page"))
            if claim:
                lines.extend(["", "## Evidence", "", f"> {claim}", "", f"Pages: {pages}"])

    return "\n".join(lines).strip() + "\n"


def _slides_html(record: dict, analysis_id: str) -> str:
    result = record.get("result", {})
    script = result.get("video_script")
    if not isinstance(script, dict) or not isinstance(script.get("scenes"), list) or not script["scenes"]:
        raise HTTPException(status_code=404, detail="Video script not found. Generate a script before downloading slides.")

    title = _markdown_text(script.get("title") or result.get("paper_title") or Path(record.get("filename") or "Slides").stem)
    slides: list[str] = []
    cover_meta = f"{len(script['scenes'])} slides"
    slides.append(f"""
      <section class="slide cover">
        <div>
          <p class="kicker">DeepDoc Slides</p>
          <h1>{html.escape(title)}</h1>
          <p class="meta">{html.escape(cover_meta)}</p>
        </div>
      </section>
    """)

    for scene in script["scenes"]:
        if not isinstance(scene, dict):
            continue
        heading = _markdown_text(scene.get("heading") or f"Slide {scene.get('scene_number', '')}".strip())
        raw_bullets = scene.get("bullets")
        bullets = raw_bullets if isinstance(raw_bullets, list) else []
        bullet_items = []
        for bullet in bullets:
            bullet_text = _markdown_text(bullet)
            if bullet_text:
                bullet_items.append(f"<li>{html.escape(bullet_text)}</li>")
        bullet_html = "\n".join(bullet_items)
        voiceover = _markdown_text(scene.get("voiceover"))
        evidence = scene.get("evidence")
        evidence_html = ""
        if isinstance(evidence, dict):
            claim = _markdown_text(evidence.get("claim") or evidence.get("excerpt") or evidence.get("summary"))
            pages = _format_pages(evidence.get("pages") or evidence.get("page_numbers") or evidence.get("page_number") or evidence.get("page"))
            if claim:
                evidence_html = f"""
                  <aside class="evidence">
                    <strong>Evidence</strong>
                    <p>{html.escape(claim)}</p>
                    <span>Pages {html.escape(pages)}</span>
                  </aside>
                """
        notes_html = f"<p class=\"notes\">{html.escape(voiceover)}</p>" if voiceover else ""
        slides.append(f"""
          <section class="slide">
            <div class="slide-copy">
              <p class="kicker">{html.escape(_markdown_text(scene.get("role") or "Slide"))}</p>
              <h1>{html.escape(heading)}</h1>
              <ul>{bullet_html}</ul>
            </div>
            {evidence_html}
            {notes_html}
          </section>
        """)

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #1d1d1f;
        --muted: #6e6e73;
        --accent: #0071e3;
        --line: rgba(0, 0, 0, 0.1);
        --surface: #f5f5f7;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--surface);
        color: var(--ink);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
      }}
      .deck {{
        display: grid;
        gap: 28px;
        padding: 28px;
      }}
      .slide {{
        position: relative;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: min(720px, calc(100vh - 56px));
        aspect-ratio: 16 / 9;
        max-width: 1280px;
        width: min(100%, 1280px);
        margin: 0 auto;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: clamp(48px, 7vw, 86px);
        background: #fff;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.12);
        overflow: hidden;
        page-break-after: always;
      }}
      .cover {{
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
      }}
      .kicker {{
        margin: 0 0 18px;
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 760;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }}
      h1 {{
        max-width: 980px;
        margin: 0;
        font-size: clamp(2.8rem, 6vw, 5.6rem);
        line-height: 0.98;
        letter-spacing: 0;
      }}
      .meta {{
        margin-top: 26px;
        color: var(--muted);
        font-size: 1.2rem;
        font-weight: 650;
      }}
      ul {{
        display: grid;
        gap: 16px;
        max-width: 720px;
        margin: 34px 0 0;
        padding: 0;
        list-style: none;
      }}
      li {{
        border-left: 4px solid var(--accent);
        padding-left: 18px;
        font-size: clamp(1.35rem, 2.4vw, 2.1rem);
        font-weight: 720;
        line-height: 1.16;
      }}
      .evidence {{
        margin-top: 36px;
        max-width: 760px;
        border-top: 1px solid var(--line);
        padding-top: 18px;
        color: var(--muted);
      }}
      .evidence strong {{
        color: var(--ink);
        font-size: 0.9rem;
      }}
      .evidence p {{
        margin: 8px 0;
        font-size: 1rem;
        line-height: 1.45;
      }}
      .notes {{
        position: absolute;
        left: -9999px;
      }}
      @media print {{
        body {{ background: #fff; }}
        .deck {{ gap: 0; padding: 0; }}
        .slide {{
          width: 100vw;
          height: 100vh;
          max-width: none;
          min-height: 0;
          border: 0;
          border-radius: 0;
          box-shadow: none;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="deck">
      {"".join(slides)}
    </main>
  </body>
</html>
"""


def _record_pdf_path(record: dict) -> Path:
    source_pdf_path = str(record.get("source_pdf_path") or "").strip()
    if source_pdf_path:
        stored_path = Path(source_pdf_path)
        if stored_path.exists():
            return stored_path
        if not stored_path.is_absolute():
            service_relative = BASE_DIR / stored_path
            if service_relative.exists():
                return service_relative

    filename = Path(record.get("filename") or "").name
    if not filename:
        return Path()

    candidates = [
        UPLOAD_DIR / filename,
        BASE_DIR / "uploads" / filename,
        BASE_DIR.parent / "uploads" / filename,
        Path("uploads") / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check():
    gemini_ready = is_gemini_configured()
    ffmpeg_ready = shutil.which("ffmpeg") is not None
    tts_health = _tts_health()
    mp4_ready = ffmpeg_ready and bool(tts_health.get("ready"))
    return {
        "status": "ok" if gemini_ready and mp4_ready else "degraded",
        "gemini_configured": gemini_ready,
        "ffmpeg_available": ffmpeg_ready,
        "tts": tts_health,
        "mp4_ready": mp4_ready,
        "upload_dir": str(UPLOAD_DIR),
        "static_dir": str(STATIC_DIR),
    }


@app.post("/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    summary_mode: str = Form("standard"),
):
    _ensure_gemini_configured()

    filename = Path(file.filename or "uploaded.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    stored_filename = _safe_pdf_storage_name(filename)
    file_path = UPLOAD_DIR / stored_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        return analyze_pdf_file(file_path, filename, summary_mode=summary_mode)
    except Exception:
        _remove_file_if_exists(file_path)
        raise


@app.get("/arxiv/search")
async def arxiv_search(
    q: str = Query(..., min_length=1, max_length=160),
    max_results: int = Query(10, ge=1, le=25),
):
    try:
        return {"results": search_arxiv(q, max_results=max_results)}
    except ArxivServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/arxiv/analyze")
async def analyze_arxiv_paper(request: ArxivAnalyzeRequest):
    _ensure_gemini_configured()

    if not is_valid_arxiv_pdf_url(request.pdf_url):
        raise HTTPException(status_code=400, detail="Invalid arXiv PDF URL.")

    url_arxiv_id = arxiv_id_from_pdf_url(request.pdf_url)
    if not _arxiv_ids_match(url_arxiv_id or "", request.arxiv_id):
        raise HTTPException(status_code=400, detail="The arXiv ID does not match the PDF URL.")

    try:
        storage_arxiv_id = f"{request.arxiv_id}-{uuid4().hex[:12]}"
        pdf_path = download_arxiv_pdf(
            request.pdf_url,
            storage_arxiv_id,
            target_dir=UPLOAD_DIR,
        )
    except ArxivServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        return analyze_pdf_file(
            pdf_path,
            f"arxiv-{request.arxiv_id}.pdf",
            summary_mode=request.summary_mode,
        )
    except HTTPException:
        _remove_file_if_exists(pdf_path)
        raise
    except Exception as error:
        _remove_file_if_exists(pdf_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}") from error


@app.get("/analyses")
async def get_analyses():
    return {"analyses": list_analyses()}


@app.post("/analyses/{analysis_id}/reanalyze")
async def reanalyze_existing_pdf(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    pdf_path = _record_pdf_path(record)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Original PDF not found. Upload or analyze the paper again.")

    result = record.get("result", {})
    summary_mode = result.get("summary_mode") or "standard"
    return analyze_pdf_file(
        pdf_path,
        Path(record.get("filename") or pdf_path.name).name,
        summary_mode=summary_mode,
    )


@app.post("/analyses/{analysis_id}/video-script")
async def create_video_script(
    analysis_id: str,
    slide_count: int = Query(10, ge=3, le=20),
):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if not is_gemini_configured():
        raise HTTPException(status_code=503, detail=gemini_configuration_error())

    result = record.get("result", {})
    video_script = generate_video_script(result, slide_count=slide_count)
    updated_record = save_video_script(analysis_id, video_script)
    if not updated_record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": analysis_id,
        "video_script": video_script,
    }


@app.post("/analyses/{analysis_id}/video")
async def create_video(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    result = record.get("result", {})
    video_script = result.get("video_script")
    if not video_script:
        raise HTTPException(
            status_code=400,
            detail="Generate a video script before generating a video.",
        )

    try:
        summary = result.get("document_summary", {})
        display_title = (
            result.get("paper_title")
            or summary.get("title")
            or Path(record.get("filename") or "Research explainer").stem
        )
        video_result = generate_video_from_script(analysis_id, video_script, display_title=display_title)
    except VideoGenerationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    updated_record = save_video_result(analysis_id, video_result)
    if not updated_record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": analysis_id,
        "video": video_result,
    }


@app.get("/analyses/{analysis_id}/video/download")
async def download_video(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    video = record.get("result", {}).get("video", {})
    video_path_value = video.get("video_path") if isinstance(video, dict) else ""
    if not isinstance(video_path_value, str) or not video_path_value:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video_path_value)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=_artifact_filename(record, analysis_id, "video", "mp4"),
    )


@app.get("/analyses/{analysis_id}/video-script/download")
async def download_video_script(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    video_script = record.get("result", {}).get("video_script")
    if not video_script:
        raise HTTPException(status_code=404, detail="Video script not found")

    content = json.dumps(video_script, ensure_ascii=False, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{_artifact_filename(record, analysis_id, "video-script", "json")}"'
            )
        },
    )


@app.get("/analyses/{analysis_id}/markdown/download")
async def download_analysis_markdown(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    content = _analysis_markdown(record, analysis_id)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{_artifact_filename(record, analysis_id, "analysis", "md")}"'
            )
        },
    )


@app.get("/analyses/{analysis_id}/slides/download")
async def download_slides_markdown(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    content = _slides_markdown(record, analysis_id)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{_artifact_filename(record, analysis_id, "slides", "md")}"'
            )
        },
    )


@app.get("/analyses/{analysis_id}/slides-html/download")
async def download_slides_html(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    content = _slides_html(record, analysis_id)
    return Response(
        content=content,
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{_artifact_filename(record, analysis_id, "slides-html", "html")}"'
            )
        },
    )


@app.head("/analyses/{analysis_id}/pdf")
@app.get("/analyses/{analysis_id}/pdf")
async def get_analysis_pdf(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    pdf_path = _record_pdf_path(record)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Original PDF not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=_artifact_filename(record, analysis_id, "source-pdf", "pdf"),
        content_disposition_type="inline",
    )


@app.get("/analyses/{analysis_id}/download")
async def download_analysis(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return FileResponse(
        Path(__file__).resolve().parent / "data" / "analyses" / f"{analysis_id}.json",
        media_type="application/json",
        filename=_artifact_filename(record, analysis_id, "analysis", "json"),
    )


@app.delete("/analyses/{analysis_id}")
async def delete_saved_analysis(
    analysis_id: str,
    delete_files: bool = Query(True, description="Delete uploaded PDFs and generated video files with the history record."),
):
    deletion_result = delete_analysis(analysis_id, delete_files=delete_files)
    if not deletion_result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return deletion_result


@app.get("/analyses/{analysis_id}")
async def get_analysis_by_id(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return record
