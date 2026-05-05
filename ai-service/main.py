from fastapi import FastAPI, UploadFile, File
from datetime import datetime, timezone
import os
import shutil
import time
from pathlib import Path

from pipeline import run_pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    started_at = time.perf_counter()
    submitted_at = datetime.now(timezone.utc)

    filename = Path(file.filename or "uploaded.pdf").name
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = run_pipeline(file_path)
    generated_at = datetime.now(timezone.utc)
    result["submitted_at"] = submitted_at.isoformat()
    result["generated_at"] = generated_at.isoformat()
    result["processing_seconds"] = round(time.perf_counter() - started_at, 2)

    return result
