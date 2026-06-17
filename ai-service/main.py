from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
import os
import shutil
import time
from pathlib import Path

from llm.summarizer import generate_video_script
from pipeline import run_pipeline
from storage import get_analysis, list_analyses, save_analysis, save_video_result, save_video_script
from video_generator import VideoGenerationError, generate_video_from_script

app = FastAPI(
    title="DeepDoc",
    description="AI-powered research paper analysis service.",
)

UPLOAD_DIR = "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    summary_mode: str = Form("standard"),
):
    started_at = time.perf_counter()
    submitted_at = datetime.now(timezone.utc)

    filename = Path(file.filename or "uploaded.pdf").name
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = run_pipeline(file_path, summary_mode=summary_mode)
    generated_at = datetime.now(timezone.utc)
    result["submitted_at"] = submitted_at.isoformat()
    result["generated_at"] = generated_at.isoformat()
    result["processing_seconds"] = round(time.perf_counter() - started_at, 2)
    record = save_analysis(filename, result)

    return record["result"]


@app.get("/analyses")
async def get_analyses():
    return {"analyses": list_analyses()}


@app.post("/analyses/{analysis_id}/video-script")
async def create_video_script(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    result = record.get("result", {})
    video_script = generate_video_script(result)
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
    video_path = Path(video.get("video_path", ""))
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    filename = Path(record.get("filename") or "analysis").stem
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{filename}-{analysis_id}.mp4",
    )


@app.get("/analyses/{analysis_id}/download")
async def download_analysis(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    filename = Path(record.get("filename") or "analysis").stem
    return FileResponse(
        Path(__file__).resolve().parent / "data" / "analyses" / f"{analysis_id}.json",
        media_type="application/json",
        filename=f"{filename}-{analysis_id}.json",
    )


@app.get("/analyses/{analysis_id}")
async def get_analysis_by_id(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return record
