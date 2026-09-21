# DeepDoc

DeepDoc is an AI-powered research workspace for reading and understanding academic papers. Upload PDFs or analyze arXiv papers, read the original PDF side by side with an evidence-grounded AI brief, and generate references, slide outlines, and optional narrated videos.

DeepDoc is organized around projects:

- A **Project** is the workspace container.
- A **Paper** is one resource inside a project.
- A single-paper analysis is simply a project with one paper.
- Multi-paper projects use the same workspace layout and add lightweight paper navigation.

## Features

- Upload one or multiple PDFs.
- Analyze arXiv papers by search or direct arXiv ID / URL.
- Use one unified workspace for single-paper and multi-paper reading.
- View AI understanding and the original PDF side by side.
- Overview, Evidence, References, and Slides tabs.
- Keep signed-in history isolated by Supabase user.
- Store signed-in source PDFs and generated artifacts in Cloudflare R2.
- Let anonymous visitors analyze papers in browser-only mode.
- Reanalyze a paper in another mode without overwriting the existing analysis.

## Stack

- FastAPI
- Static HTML/CSS/JS
- Gemini or a local Qwen model through Ollama
- Supabase Auth + Postgres
- Cloudflare R2
- PyMuPDF
- Optional Piper + ffmpeg for narrated MP4 generation

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3 -m venv ai-service/venv
ai-service/venv/bin/python -m pip install -r requirements.txt
```

Gemini is the default provider:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
```

To use a local Qwen model through Ollama instead:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
DEEPDOC_OLLAMA_MODELS=qwen3:8b,qwen3:4b
OLLAMA_CONTEXT_LENGTH=12288
OLLAMA_NUM_PREDICT=3072
OLLAMA_THINK=false
OLLAMA_KEEP_ALIVE=10m
OLLAMA_CPU_ONLY=false
```

Ollama uses the available accelerator by default. Set `OLLAMA_CPU_ONLY=true`
only when running a CPU/VPS benchmark. Start Ollama before DeepDoc with
`ollama serve`. On a lab server, point
`OLLAMA_BASE_URL` to the Ollama service's private network address. A
comma-separated `DEEPDOC_OLLAMA_MODELS` value can define selectable models.
`OLLAMA_MODELS` remains reserved for Ollama's model storage directory.

```bash
cd ai-service
venv/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- App: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

For Supabase, R2, Render, and MP4 setup, see [Deployment Guide](docs/deployment.md).

## Testing

```bash
PYTHONPYCACHEPREFIX=/private/tmp/deepdoc-pyc ai-service/venv/bin/python -m unittest ai-service/test_projects.py ai-service/test_evaluator.py
PYTHONPYCACHEPREFIX=/private/tmp/deepdoc-pyc ai-service/venv/bin/python -m py_compile ai-service/storage.py ai-service/main.py ai-service/utils/summary_prompt.py ai-service/llm/evaluator.py ai-service/test_projects.py ai-service/test_evaluator.py
node --check ai-service/static/app.js
git diff --check
```

## Summary evaluation

`llm.evaluator.SummaryEvaluator` compares summaries from one or more named
models against evidence packets. Gemini is the default judge; pass any object
with an `evaluate(summary, evidence_packet)` method to replace it with another
remote or local judge.

```python
from llm.evaluator import SummaryEvaluator

evaluator = SummaryEvaluator(
    models={
        "gemini": gemini_result,       # string or {"summary": "..."}
        "local-model": local_result,
    },
    evidence_packets=[
        {"id": "method", "text": method_excerpt},
        {"id": "results", "text": results_excerpt},
    ],
)

scores = evaluator.evaluate("default")
details_json = evaluator.evaluate_json("detailed", indent=2)
averages = evaluator.evaluate("average")
averages_with_chart = evaluator.evaluate("average", visualize=True)
```

If `evidence_packets` is omitted, the editable benchmark packets in
`ai-service/evaluation/default_packets.json` are used. For model names without
pre-generated summaries, pass `summary_generator(model_name, prompt)`. Set
`GEMINI_EVALUATOR_MODEL` to change the Gemini judge model without a code change.

Callable models receive the exact prompt produced by
`utils.summary_prompt.build_research_summary_prompt`; choose its Quick,
Standard, or Detailed variant with the evaluator's `summary_mode` argument.
Average mode returns one set of mean scores per model. With `visualize=True`,
the evaluator saves a PNG comparison chart under `ai-service/data/eval/` and
returns its location in `visualization_path`. Pass `evaluation_dir` to use a
different output directory.

Default and detailed packet-level results are keyed by model name or ID:

```json
{
  "model-a": [
    {"packet_id": "method", "evaluation": {}}
  ],
  "model-b": [
    {"packet_id": "method", "evaluation": {}}
  ]
}
```

`ai-service/test_models.py` calls Gemini to list models and is an external API probe, not a normal local regression test.

## Runtime Data

Generated runtime files are ignored by git.

- Uploaded/source PDFs: `ai-service/uploads/`
- Local JSON analysis records: `ai-service/data/analyses/`
- Generated videos: `ai-service/data/videos/`
- Evaluation charts: `ai-service/data/eval/`
- Regression outputs: `ai-service/data/eval_*`

## Documentation

- [Deployment Guide](docs/deployment.md)
- [Supabase schema](supabase/schema.sql)
