from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
import os
import shutil
import time
from pathlib import Path

from pipeline import run_pipeline
from storage import get_analysis, list_analyses, save_analysis

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
