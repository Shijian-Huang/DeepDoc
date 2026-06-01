from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
os.makedirs(UPLOAD_DIR, exist_ok=True)

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


@app.get("/analyses/{analysis_id}")
async def get_analysis_by_id(analysis_id: str):
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return record
