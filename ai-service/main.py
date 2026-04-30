from fastapi import FastAPI, UploadFile, File
import os
import shutil
from pathlib import Path

from pipeline import run_pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "uploaded.pdf").name
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = run_pipeline(file_path)

    return result