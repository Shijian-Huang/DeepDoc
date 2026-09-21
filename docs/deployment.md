# Deployment Guide

This guide covers the production configuration for DeepDoc: Supabase, Cloudflare R2, Render, and optional MP4 generation.

## Environment Variables

DeepDoc defaults to a local Qwen model served by Ollama:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
DEEPDOC_OLLAMA_MODELS=qwen3:8b,qwen3:4b
OLLAMA_CONTEXT_LENGTH=12288
OLLAMA_NUM_PREDICT=3072
OLLAMA_THINK=false
OLLAMA_KEEP_ALIVE=10m
OLLAMA_CPU_ONLY=false
```

`OLLAMA_CPU_ONLY=false` allows Ollama to use an available GPU (including Metal
on macOS). Set it to `true` only for a controlled CPU-only benchmark.

`OLLAMA_MODELS` is reserved by Ollama for its model storage directory. On WAVE,
set it to a project path such as `/WAVE/projects2/DeepDoc/ollama-models`.

To use Gemini instead, set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`.

For a separate inference server, set `OLLAMA_BASE_URL` to its private network
address and configure Ollama to listen on that interface. Do not expose an
unauthenticated Ollama endpoint directly to the public internet.

Supabase Auth + Postgres:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_your_publishable_key
SUPABASE_SECRET_KEY=sb_secret_your_secret_key
```

Legacy compatibility names are still accepted:

```bash
SUPABASE_ANON_KEY=your_legacy_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_legacy_service_role_key
```

`SUPABASE_SECRET_KEY` or `SUPABASE_SERVICE_ROLE_KEY` must only be set on the backend or Render environment. Never expose it in browser code.

Cloudflare R2:

```bash
R2_BUCKET=deepdoc-files
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_REGION=auto
```

## Supabase Schema

Run the schema before deploying project-enabled code:

1. Open `supabase/schema.sql`.
2. Copy the SQL content.
3. Paste it into Supabase SQL Editor.
4. Run it.
5. Then run:

```sql
NOTIFY pgrst, 'reload schema';
```

The schema creates:

- `public.projects`
- `public.analyses`
- `project_id`, `status`, and `updated_at` support on analyses
- indexes for project/history queries
- RLS policies for authenticated users

Supabase may warn that the query contains destructive operations because the file drops and recreates RLS policies. It does not drop tables or delete analysis data.

If Render logs show:

```text
Could not find the table 'public.projects' in the schema cache
```

run `supabase/schema.sql`, then reload the schema cache with `NOTIFY pgrst, 'reload schema';`.

## Storage Behavior

Anonymous users:

- Can analyze papers without signing in.
- Store project metadata in browser localStorage.
- Store uploaded PDFs in IndexedDB when available.
- Can reopen browser-only projects on the same browser.
- Cannot see those projects across devices or after clearing site data.

Signed-in users:

- Store projects and analyses in Supabase.
- Store source PDFs in R2 when R2 is configured.
- Access is filtered by verified Supabase user ID.

New signed-in uploads are stored with paths shaped like:

```text
users/{user_id}/projects/{project_id}/analyses/{analysis_id}/original.pdf
```

Legacy paths continue to work when present.

## Optional MP4 Generation

DeepDoc can generate narrated MP4 videos from slide outlines.

Requirements:

- `ffmpeg` installed on the server
- Piper binary available
- Piper voice model configured

The `/health` endpoint reports:

- `ffmpeg_available`
- `tts.ready`
- `tts.piper_binary_available`
- `tts.piper_model_available`
- `mp4_ready`

MP4 generation is optional. Summary, evidence, references, PDF reading, and slide outlines can work without it.

## Render Checklist

Before deploying:

- Set `GEMINI_API_KEY`, or configure `LLM_PROVIDER=ollama` and start Ollama.
- If using login/history, set Supabase URL and keys.
- Run `supabase/schema.sql` in Supabase.
- If using persistent source PDFs, set R2 variables.
- Confirm the server can write to `ai-service/uploads/` and `ai-service/data/`.
- Install `ffmpeg` if MP4 generation is enabled.

After deploying:

- Check `/health`.
- Confirm `llm_configured: true` and that `llm_provider` is correct.
- Confirm `storage_backend`, `supabase_auth_enabled`, and `r2_storage_enabled` match the intended setup.
- If MP4 is enabled, confirm `mp4_ready: true`.

After changing static JS/CSS:

- Update `APP_VERSION` in `ai-service/main.py`.
- Update the script cache-buster in `ai-service/static/index.html`.

## Manual End-to-End Checklist

After a deploy, verify:

1. Anonymous user can upload one PDF and see the PDF side by side.
2. Anonymous user can refresh and reopen the local project on the same browser.
3. Signed-in user can upload one PDF and see it in History.
4. Signed-in user can upload multiple PDFs and switch papers.
5. Project name is editable and separate from paper title.
6. Reanalyze opens a confirmation dialog and creates a new analysis.
7. User A cannot see User B's projects.
8. R2-backed PDFs reopen after refresh and redeploy.
9. Direct arXiv analysis works by ID, abs URL, or PDF URL.

## Key API Routes

- `POST /analyze-pdf`
- `GET /arxiv/search`
- `POST /arxiv/analyze`
- `GET /projects`
- `GET /projects/{project_id}`
- `PATCH /projects/{project_id}`
- `DELETE /projects/{project_id}`
- `POST /projects/analyze-pdfs`
- `POST /projects/{project_id}/papers`
- `GET /analyses/{analysis_id}`
- `POST /analyses/{analysis_id}/reanalyze`
- `GET /analyses/{analysis_id}/pdf`
- `GET /analyses/{analysis_id}/download`
- `DELETE /analyses/{analysis_id}`
