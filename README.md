# DeepDoc

DeepDoc is a research-paper workspace for uploading or searching arXiv papers, generating structured analyses with evidence, viewing the original PDF side by side, and exporting analysis, slides, and optional narrated MP4 videos.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv ai-service/venv
ai-service/venv/bin/python -m pip install -r requirements.txt
```

Create `ai-service/.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key
```

Optional Supabase login + Postgres storage:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_your_publishable_key
SUPABASE_SECRET_KEY=sb_secret_your_secret_key
```

Open `supabase/schema.sql`, copy its SQL content, paste it into the Supabase SQL Editor, and run it before enabling these variables. Do not paste the file path itself into the SQL Editor. `SUPABASE_SECRET_KEY` must only be set on the backend or Render environment; never expose it in browser code. Legacy `SUPABASE_ANON_KEY` and `SUPABASE_SERVICE_ROLE_KEY` are still accepted for compatibility.

When Supabase is configured, anonymous visitors can still analyze papers, but those results are saved only in the current browser's local storage. Signed-in users store analysis records in Supabase with that user's verified `user_id`, so the same account can see its history across devices. Without Supabase variables, DeepDoc keeps the existing local JSON storage mode.

For MP4 generation, install `ffmpeg`. DeepDoc uses Piper TTS by default; install or configure a Piper voice model, or set another TTS provider through environment variables in `ai-service/video_generator.py`.

## Run Locally

```bash
cd ai-service
venv/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- App: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Runtime Data

Generated runtime files are intentionally ignored by git:

- Uploaded/source PDFs: `ai-service/uploads/`
- Analysis records: `ai-service/data/analyses/`
- Generated videos: `ai-service/data/videos/`
- Regression outputs: `ai-service/data/eval_*`

Deleting an analysis from the UI deletes the history record and, by default, its stored source PDF and generated video files from the server. The backend also supports record-only deletion with:

```http
DELETE /analyses/{analysis_id}?delete_files=false
```

## Deployment Notes

Before deploying, verify:

- `GEMINI_API_KEY` is configured in the server environment.
- If login/history isolation is enabled, all Supabase environment variables above are configured.
- `/health` reports `gemini_configured: true`.
- If MP4 generation is enabled, `/health` also reports `mp4_ready: true`.
- The server has write access to `ai-service/uploads/` and `ai-service/data/`.
- Disk retention policy is acceptable for uploaded PDFs and generated videos.
- `ffmpeg` is installed if MP4 generation is enabled.
