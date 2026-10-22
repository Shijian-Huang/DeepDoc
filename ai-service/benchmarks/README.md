# Benchmark paper sets

Fixed papers for measuring DeepDoc's production pipeline with `tools/benchmark.py`.

| Set | Papers | Use |
|---|---|---|
| `dev.json` | 8 ACLSum test-split papers (ACL Anthology, 2016+) + 2 recent arXiv papers (astro-ph, q-bio) | Run after every pipeline change. Safe to inspect and debug. |
| `holdout.json` | 8 further ACLSum test-split papers + 2 recent arXiv papers (astro-ph, cond-mat) | Final validation only. Do not read its outputs, tune prompts on it, or show it to an AI assistant during development. |

Each paper is pinned by URL and SHA-256. PDFs are downloaded into the git-ignored `data/benchmark_papers/`, not committed. ACL papers carry ACLSum's expert reference sentences (MIT, revision `08c2433`) for the challenge, approach, and outcome aspects, plus the source sentences the annotators marked for each aspect.

Holdout ACLSum papers were not hand-picked: they are the next 8 post-2016 papers in the ACLSum test split after the dev papers. arXiv papers were chosen as the most recent in their category that are at most 25 pages and not machine-learning method papers.

## Workflow

```bash
cd ai-service
venv/bin/python tools/benchmark.py fetch --set dev
venv/bin/python tools/benchmark.py run --set dev --provider ollama --model qwen3:8b --repeats 3 --tag v1
venv/bin/python tools/benchmark.py check v1
venv/bin/python tools/benchmark.py compare v1 v2
```

`run` resumes: rerunning the same command skips outputs that already exist. A new pipeline version needs a new `--tag`.

## What the checks measure

The checks are deterministic proxies with no model calls. Use them for trends and to locate likely causes, not as final quality scores:

- numbers not found anywhere in the paper;
- near-verbatim copying;
- empty or duplicated fields;
- coverage of the authors' abstract and of ACLSum aspects (word-overlap recall);
- whether ACLSum's source sentences reached the model input;
- invalid evidence references;
- failures, timings, retries, and fallbacks.

ACLSum references come from the abstract, introduction, and conclusion, which DeepDoc almost always selects. So ACLSum shows little about content lost from results or method sections; the number and abstract checks are more sensitive to that.
