#!/usr/bin/env python3
"""Run DeepDoc's production pipeline on a fixed paper set, check the saved outputs, and compare runs.

    venv/bin/python tools/benchmark.py fetch --set dev
    venv/bin/python tools/benchmark.py run --set dev --provider ollama --model qwen3:8b --repeats 3 --tag v1
    venv/bin/python tools/benchmark.py check v1
    venv/bin/python tools/benchmark.py compare v1 v2

Paper lists live in ``benchmarks/<set>.json``. PDFs and run outputs are written under the
git-ignored ``data/`` directory. The checks are deterministic proxies (no model calls); they
show trends and point to likely causes, and are meant to be replaced by the independent
evaluator as it matures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARK_DIR = ROOT / "benchmarks"
PAPER_DIR = ROOT / "data" / "benchmark_papers"
RUN_DIR = ROOT / "data" / "benchmark_runs"
USER_AGENT = "DeepDoc/0.1 (benchmark runner)"

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "their", "they", "them", "these", "those",
    "are", "was", "were", "has", "have", "had", "been", "its", "our", "can", "also", "such", "than", "which",
    "while", "using", "used", "use", "based", "paper", "study", "work", "propose", "proposed", "show", "shows",
    "results", "result", "method", "approach", "model", "models", "new", "both", "more", "most", "not", "but",
}
NUMBER_PATTERN = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d])|(?<![\w.])\d+(?:\.\d+)?")
EVALUATIVE_PATTERN = re.compile(
    r"(?<![a-z0-9])(novel|first|significant|significantly|best|state-of-the-art|breakthrough|groundbreaking)(?![a-z0-9])",
    re.IGNORECASE,
)
COVERAGE_THRESHOLD = 0.5
COPY_SHINGLE = 5
COPY_THRESHOLD = 0.8
DUPLICATE_THRESHOLD = 0.6


class BenchmarkError(RuntimeError):
    """Raised for invalid benchmark input or state."""


# ---------------------------------------------------------------- paper sets

def load_paper_set(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = BENCHMARK_DIR / f"{name_or_path}.json"
    try:
        paper_set = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Could not load paper set {path}: {error}") from error
    paper_set["_path"] = str(path)
    paper_set["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return paper_set


def paper_pdf_path(paper: dict[str, Any], paper_dir: Path = PAPER_DIR) -> Path:
    return paper_dir / f"{paper['id'].replace(':', '-').replace('/', '_')}.pdf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_papers(paper_set: dict[str, Any], paper_dir: Path = PAPER_DIR, pause_seconds: float = 3.0) -> list[str]:
    """Download missing PDFs and verify every checksum. Returns problems (empty when all good)."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for paper in paper_set["papers"]:
        target = paper_pdf_path(paper, paper_dir)
        if not target.exists():
            partial = target.with_suffix(".part")
            request = urllib.request.Request(paper["pdf_url"], headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    partial.write_bytes(response.read())
            except OSError as error:
                problems.append(f"{paper['id']}: download failed: {error}")
                continue
            partial.replace(target)
            print(f"downloaded {paper['id']}")
            time.sleep(pause_seconds)
        digest = sha256_file(target)
        if digest != paper["sha256"]:
            problems.append(f"{paper['id']}: SHA-256 {digest} does not match the pinned {paper['sha256']}")
    return problems


# ------------------------------------------------------------------ running

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_manifest(paper_set: dict[str, Any], tag: str, provider: str, model: str, summary_mode: str, repeats: int) -> dict[str, Any]:
    from llm import summarizer

    settings: dict[str, Any] = {"provider": provider, "model": model, "summary_mode": summary_mode}
    if provider == "ollama":
        settings.update({
            "temperature": 0.1,
            "schema_enforced": True,
            "context_length": summarizer.ollama_context_length,
            "num_predict": summarizer.ollama_num_predict,
            "think": summarizer.ollama_think,
            "cpu_only": summarizer.ollama_cpu_only,
        })
    else:
        settings.update({"temperature": "provider default", "schema_enforced": False})
    return {
        "tag": tag,
        "paper_set": paper_set["name"],
        "paper_set_sha256": paper_set["_sha256"],
        "paper_ids": [paper["id"] for paper in paper_set["papers"]],
        "repeats": repeats,
        "settings": settings,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def config_name(provider: str, model: str) -> str:
    return f"{provider}-{re.sub(r'[^A-Za-z0-9.]+', '-', model).strip('-')}"


def run_benchmark(
    paper_set: dict[str, Any],
    tag: str,
    provider: str,
    model: str,
    repeats: int = 3,
    summary_mode: str | None = None,
    paper_dir: Path = PAPER_DIR,
    run_dir: Path = RUN_DIR,
) -> Path:
    from llm.summarizer import use_llm_selection
    from pipeline import run_pipeline

    summary_mode = summary_mode or paper_set.get("summary_mode", "standard")
    missing = [paper["id"] for paper in paper_set["papers"] if not paper_pdf_path(paper, paper_dir).exists()]
    if missing:
        raise BenchmarkError(f"Missing PDFs ({', '.join(missing)}). Run: tools/benchmark.py fetch --set {paper_set['name']}")

    try:
        with use_llm_selection(provider, model):
            pass
    except (ValueError, RuntimeError) as error:
        raise BenchmarkError(
            f"{provider}/{model} is not available: {error}. For Ollama, check that it is running at "
            "OLLAMA_BASE_URL and that the model has been pulled."
        ) from error

    output_dir = run_dir / tag / config_name(provider, model)
    manifest_path = output_dir / "manifest.json"
    manifest = build_manifest(paper_set, tag, provider, model, summary_mode, repeats)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("paper_set_sha256", "settings", "git_sha", "repeats"):
            if previous.get(key) != manifest[key]:
                raise BenchmarkError(
                    f"{output_dir} already holds a run with different {key}. Use a new --tag instead of mixing runs."
                )
        manifest["started_at"] = previous.get("started_at", manifest["started_at"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total =len(paper_set["papers"]) * repeats
    done = 0
    for paper in paper_set["papers"]:
        paper_output = output_dir / paper["id"].replace(":", "-")
        paper_output.mkdir(exist_ok=True)
        for repeat in range(1, repeats + 1):
            done += 1
            target = paper_output / f"run-{repeat}.json"
            if target.exists():
                print(f"[{done}/{total}] {paper['id']} run {repeat}: already saved, skipping")
                continue
            started = time.perf_counter()
            record: dict[str, Any] = {"paper_id": paper["id"], "repeat": repeat}
            try:
                with use_llm_selection(provider, model):
                    result = run_pipeline(str(paper_pdf_path(paper, paper_dir)), summary_mode=summary_mode)
                record.update({"status": "ok", "result": result})
            except KeyboardInterrupt:
                raise
            except Exception as error:  # a failed paper must not stop the benchmark
                record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
            record["wall_seconds"] = round(time.perf_counter() - started, 2)
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[{done}/{total}] {paper['id']} run {repeat}: {record['status']} in {record['wall_seconds']}s")

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir


# ------------------------------------------------------------------- checks

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def content_tokens(text: str) -> set[str]:
    return {word for word in _words(text) if len(word) >= 3 and word not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", _normalize(text))
    return [part.strip() for part in parts if len(part.strip().split()) >= 4]


def extract_numbers(text: str) -> set[float]:
    return {float(match.replace(",", "")) for match in NUMBER_PATTERN.findall(str(text or ""))}


def token_recall(reference: str, candidate: str) -> float:
    reference_tokens = content_tokens(reference)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & content_tokens(candidate)) / len(reference_tokens)


def best_recall(reference: str, candidates: list[str]) -> float:
    return max((token_recall(reference, candidate) for candidate in candidates), default=0.0)


def _shingles(words: list[str], size: int = COPY_SHINGLE) -> set[tuple[str, ...]]:
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


def copied_fraction(sentence: str, source_shingles: set[tuple[str, ...]]) -> float:
    sentence_shingles = _shingles(_words(sentence))
    if not sentence_shingles:
        return 0.0
    return len(sentence_shingles & source_shingles) / len(sentence_shingles)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def contains_sentence(haystack_compact: str, sentence: str) -> bool:
    """Match a (possibly tokenized) sentence against text, ignoring spacing and punctuation."""
    needle = _compact(sentence)
    if len(needle) < 20:
        return False
    return needle[:120] in haystack_compact


def _clean_abstract(text: str, limit: int = 12) -> list[str]:
    sentences = []
    for sentence in split_sentences(text):
        if re.match(r"^[∗*†‡]|equal contribution|corresponding author", sentence, re.IGNORECASE):
            break
        sentences.append(sentence)
    return sentences[:limit]


class PaperText:
    """Full text of one benchmark paper, parsed the same way DeepDoc parses it."""

    def __init__(self, pdf_path: Path):
        from parser.pdf_parser import parse_pdf_pages
        from utils.section_extractor import split_pages_into_sections

        pages = parse_pdf_pages(str(pdf_path))
        self.page_count = len(pages)
        text = " ".join(page["text"] + " " + page.get("raw_text", "") for page in pages)
        self.numbers = extract_numbers(text)
        self.shingles = _shingles(_words(text))
        self.compact = _compact(text)
        self.words = set(_words(text))
        abstract = split_pages_into_sections(pages).get("abstract", {}).get("text", "")
        self.abstract_sentences = _clean_abstract(abstract)

    def has_term(self, term: str) -> bool:
        return term in self.words if term.isalpha() else _compact(term) in self.compact


def _issue(paper_id: str, repeat: int, category: str, field: str, text: str, severity: str,
           confidence: str, component: str, reference: str = "") -> dict[str, Any]:
    return {"paper_id": paper_id, "repeat": repeat, "category": category, "field": field, "text": text,
            "severity": severity, "confidence": confidence, "component": component, "reference": reference}


def check_run(record: dict[str, Any], paper: dict[str, Any], paper_text: PaperText) -> dict[str, Any]:
    paper_id, repeat = record["paper_id"], record["repeat"]
    if record.get("status") != "ok":
        return {"paper_id": paper_id, "repeat": repeat, "failed": True, "error": record.get("error"),
                "wall_seconds": record.get("wall_seconds"),
                "issues": [_issue(paper_id, repeat, "pipeline_failure", "run", str(record.get("error")),
                                  "high", "high", "provider/config or pipeline")]}

    result = record["result"]
    summary = result.get("document_summary") or {}
    diagnostics = summary.get("diagnostics") or {}
    issues: list[dict[str, Any]] = []
    summary_sentences = split_sentences(summary.get("summary", ""))
    key_ideas = [_normalize(item) for item in summary.get("key_ideas") or [] if _normalize(item)]
    contributions = [_normalize(item) for item in summary.get("contributions") or [] if _normalize(item)]
    evidence = [item for item in summary.get("evidence") or [] if isinstance(item, dict)]
    written = [("summary", text) for text in summary_sentences] + [("key_ideas", text) for text in key_ideas] + \
              [("contributions", text) for text in contributions]
    visible = written + [("evidence", _normalize(item.get("claim"))) for item in evidence]
    written_texts = [text for _, text in written]

    # Numbers that appear nowhere in the paper.
    number_errors = 0
    for field, text in visible:
        missing = sorted(number for number in extract_numbers(text) if number not in paper_text.numbers)
        if missing:
            number_errors += 1
            issues.append(_issue(paper_id, repeat, "number_not_in_paper", field, text, "high", "medium",
                                 "generation or fact extraction",
                                 f"not found in the paper: {', '.join(f'{number:g}' for number in missing)}"))

    # Near-verbatim copying from the paper.
    copied = [(field, text) for field, text in written if copied_fraction(text, paper_text.shingles) >= COPY_THRESHOLD]
    for field, text in copied:
        issues.append(_issue(paper_id, repeat, "copied_from_paper", field, text, "low", "high", "post-processing or grounding"))
    # Evaluative wording the paper itself never uses.
    unsupported_evaluative = 0
    for field, text in written:
        terms = sorted({match.lower() for match in EVALUATIVE_PATTERN.findall(text)})
        missing = [term for term in terms if not paper_text.has_term(term)]
        if missing:
            unsupported_evaluative += len(missing)
            issues.append(_issue(paper_id, repeat, "unsupported_evaluative_wording", field, text, "medium", "high",
                                 "generation prompt", f"not used in the paper: {', '.join(missing)}"))
    fact_texts = {_normalize(fact.get("fact")).rstrip(".").lower() for fact in summary.get("verified_facts") or []}
    padded = sum(1 for text in summary_sentences if text.rstrip(".").lower() in fact_texts)

    # Empty fields and duplicated items.
    empty_fields = [name for name, items in (("key_ideas", key_ideas), ("contributions", contributions), ("evidence", evidence)) if not items]
    for name in empty_fields:
        issues.append(_issue(paper_id, repeat, "empty_field", name, "", "medium", "high", "generation or post-processing"))
    duplicates = 0
    items = [("key_ideas", text) for text in key_ideas] + [("contributions", text) for text in contributions]
    for index, (field_a, text_a) in enumerate(items):
        for field_b, text_b in items[index + 1:]:
            tokens_a, tokens_b = content_tokens(text_a), content_tokens(text_b)
            if tokens_a and tokens_b and len(tokens_a & tokens_b) / len(tokens_a | tokens_b) >= DUPLICATE_THRESHOLD:
                duplicates += 1
                issues.append(_issue(paper_id, repeat, "duplicated_content", f"{field_a}/{field_b}", text_b, "low", "medium",
                                     "post-processing", text_a))

    # Coverage of the authors' abstract.
    abstract_covered = 0
    for sentence in paper_text.abstract_sentences:
        if best_recall(sentence, written_texts) >= COVERAGE_THRESHOLD:
            abstract_covered += 1
        else:
            issues.append(_issue(paper_id, repeat, "abstract_point_missing", "summary", "", "medium", "low",
                                 "fact extraction or generation", sentence))
    abstract_coverage = abstract_covered / len(paper_text.abstract_sentences) if paper_text.abstract_sentences else None

    # ACLSum aspect coverage, and whether the supporting sentences reached the model at all.
    input_compact = _compact(" ".join(str(source.get("excerpt", "")) for source in result.get("evidence_sources") or []))
    fact_blob = " ".join(_normalize(fact.get("fact")) for fact in summary.get("verified_facts") or [])
    aspects: dict[str, Any] = {}
    for aspect, reference in (paper.get("aclsum") or {}).items():
        if aspect == "source_sentences":
            continue
        covered = best_recall(reference, written_texts) >= COVERAGE_THRESHOLD
        source_sentences = paper["aclsum"].get("source_sentences", {}).get(aspect, [])
        matchable = [sentence for sentence in source_sentences if contains_sentence(paper_text.compact, sentence)]
        in_input = any(contains_sentence(input_compact, sentence) for sentence in matchable) if matchable else None
        aspects[aspect] = {"covered": covered, "in_model_input": in_input}
        if covered:
            continue
        if in_input is False:
            category, component = "key_content_not_in_model_input", "section selection"
        elif in_input is None:
            category, component = "key_content_missing", "unknown (reference sentences not matched in the PDF text)"
        elif token_recall(reference, fact_blob) >= COVERAGE_THRESHOLD:
            category, component = "key_content_dropped_after_extraction", "generation or grounding filter"
        else:
            category, component = "key_content_not_extracted", "fact extraction"
        issues.append(_issue(paper_id, repeat, category, aspect, "", "high", "low", component, reference))

    # Evidence references.
    sources = {str(source.get("source_id")): source for source in result.get("evidence_sources") or [] if isinstance(source, dict)}
    invalid_evidence = 0
    for item in evidence:
        cited = [sources.get(str(source_id)) for source_id in item.get("source_ids") or []]
        pages = [page for page in item.get("pages") or [] if isinstance(page, int)]
        source_pages = {page for source in cited if source for page in source.get("pages", [])}
        problem = ""
        if not cited or any(source is None for source in cited):
            problem = "cites a source that is not in the model input"
        elif any(page < 1 or page > paper_text.page_count for page in pages):
            problem = "page number outside the document"
        elif pages and not set(pages) <= source_pages:
            problem = "page numbers do not match the cited source"
        if problem:
            invalid_evidence += 1
            issues.append(_issue(paper_id, repeat, "invalid_evidence_reference", "evidence", _normalize(item.get("claim")),
                                 "medium", "high", "evidence construction", problem))

    # Sentence-level source links shown to the user.
    citations = summary.get("citations") if isinstance(summary.get("citations"), dict) else None
    cited, uncited = diagnostics.get("cited_sentence_count"), diagnostics.get("uncited_sentence_count")
    invalid_citations = 0
    for field, entries in (citations or {}).items():
        for entry in entries if isinstance(entries, list) else []:
            pages = (entry or {}).get("pages") or []
            if entry and (not pages or any(not isinstance(page, int) or page < 1 or page > paper_text.page_count for page in pages)):
                invalid_citations += 1
                issues.append(_issue(paper_id, repeat, "invalid_citation_pages", field, str(entry.get("text", "")),
                                     "medium", "high", "citation mapping", f"pages: {pages}"))

    filtered = diagnostics.get("filtered_sentences") or []
    return {
        "cited_share": round(cited / (cited + uncited), 4) if citations is not None and cited is not None and (cited + uncited) else None,
        "invalid_citation_pages": invalid_citations if citations is not None else None,
        "paper_id": paper_id,
        "repeat": repeat,
        "failed": False,
        "wall_seconds": record.get("wall_seconds"),
        "summary_words": summary.get("summary_word_count"),
        "sentences_checked": len(visible),
        "number_errors": number_errors,
        "copied_ratio": round(len(copied) / len(written), 4) if written else None,
        "summary_sentences_equal_to_fact": padded,
        "empty_fields": empty_fields,
        "duplicate_pairs": duplicates,
        "abstract_coverage": None if abstract_coverage is None else round(abstract_coverage, 4),
        "aclsum": aspects,
        "aclsum_coverage": round(sum(item["covered"] for item in aspects.values()) / len(aspects), 4) if aspects else None,
        "aclsum_not_in_input": sum(1 for item in aspects.values() if item["in_model_input"] is False),
        "invalid_evidence": invalid_evidence,
        "filtered_sentences": len(filtered),
        "filtered_share": round(len(filtered) / diagnostics["checked_sentence_count"], 4)
        if diagnostics.get("checked_sentence_count") else None,
        "contributions_empty": not contributions,
        "key_idea_count": len(key_ideas),
        "contribution_count": len(contributions),
        "unsupported_evaluative_words": unsupported_evaluative,
        "compact_retry": bool(diagnostics.get("compact_retry")),
        "deterministic_fallback": bool(diagnostics.get("deterministic_fallback")),
        "verified_fact_count": diagnostics.get("verified_fact_count"),
        "source_supplement_count": diagnostics.get("source_supplement_count"),
        "issues": issues,
    }


def resolve_run_dir(spec: str, run_dir: Path = RUN_DIR) -> Path:
    path = run_dir / spec
    if (path / "manifest.json").exists():
        return path
    configs = sorted(child for child in path.glob("*") if (child / "manifest.json").exists()) if path.is_dir() else []
    if len(configs) == 1:
        return configs[0]
    if not configs:
        raise BenchmarkError(f"No benchmark run found for '{spec}' under {run_dir}.")
    names = ", ".join(f"{spec}/{config.name}" for config in configs)
    raise BenchmarkError(f"'{spec}' contains several runs; choose one of: {names}")


def check_benchmark(output_dir: Path, paper_dir: Path = PAPER_DIR, paper_set: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    paper_set = paper_set or load_paper_set(manifest["paper_set"])
    if paper_set["_sha256"] != manifest["paper_set_sha256"]:
        print(f"warning: benchmarks/{manifest['paper_set']}.json changed since this run", file=sys.stderr)
    papers = {paper["id"]: paper for paper in paper_set["papers"]}
    runs = []
    for paper_id in manifest["paper_ids"]:
        paper_text = PaperText(paper_pdf_path(papers[paper_id], paper_dir))
        for run_file in sorted((output_dir / paper_id.replace(":", "-")).glob("run-*.json")):
            runs.append(check_run(json.loads(run_file.read_text(encoding="utf-8")), papers[paper_id], paper_text))
    checks = {"run": str(output_dir.relative_to(output_dir.parents[1])), "manifest": manifest, "runs": runs}
    (output_dir / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checks


# ------------------------------------------------------------------ compare

# (label, key, better direction, formatter)
METRICS = [
    ("Failed runs (share)", "failed", "lower", "pct"),
    ("Wall time per paper (s)", "wall_seconds", "lower", "num"),
    ("Sentences with a number not in the paper", "number_errors", "lower", "num"),
    ("Copied sentences (share)", "copied_ratio", "lower", "pct"),
    ("Summary sentences that are raw facts", "summary_sentences_equal_to_fact", "lower", "num"),
    ("Abstract points covered", "abstract_coverage", "higher", "pct"),
    ("ACLSum aspects covered", "aclsum_coverage", "higher", "pct"),
    ("ACLSum aspects missing from model input", "aclsum_not_in_input", "lower", "num"),
    ("Empty fields", "empty_field_count", "lower", "num"),
    ("Duplicated item pairs", "duplicate_pairs", "lower", "num"),
    ("Invalid evidence references", "invalid_evidence", "lower", "num"),
    ("Sentences removed by grounding filter", "filtered_sentences", "lower", "num"),
    ("Generated sentences removed by grounding (share)", "filtered_share", "lower", "pct"),
    ("Runs with no contributions (share)", "contributions_empty", "lower", "pct"),
    ("Evaluative words not used in the paper", "unsupported_evaluative_words", "lower", "num"),
    ("Sentences with a source link (share)", "cited_share", "higher", "pct"),
    ("Source links with invalid pages", "invalid_citation_pages", "lower", "num"),
    ("Summary words", "summary_words", "info", "num"),
    ("Key ideas", "key_idea_count", "info", "num"),
    ("Contributions", "contribution_count", "info", "num"),
    ("Fact-extraction retries (share)", "compact_retry", "lower", "pct"),
    ("Deterministic fallbacks (share)", "deterministic_fallback", "lower", "pct"),
]


def _metric_value(run: dict[str, Any], key: str) -> float | None:
    if key == "failed":
        return float(run["failed"])
    if run["failed"]:
        return None
    if key == "empty_field_count":
        return float(len(run.get("empty_fields") or []))
    value = run.get(key)
    if isinstance(value, bool):
        return float(value)
    return None if value is None else float(value)


def per_paper_stats(checks: dict[str, Any], key: str) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation across repeats, per paper."""
    by_paper: dict[str, list[float]] = {}
    for run in checks["runs"]:
        value = _metric_value(run, key)
        if value is not None:
            by_paper.setdefault(run["paper_id"], []).append(value)
    return {
        paper_id: (statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0)
        for paper_id, values in by_paper.items()
    }


def _format(value: float | None, style: str) -> str:
    if value is None:
        return "-"
    return f"{value:.0%}" if style == "pct" else f"{value:.2f}"


def summarize(checks: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    summary = {}
    for _label, key, _direction, _style in METRICS:
        stats = per_paper_stats(checks, key)
        summary[key] = {
            "mean": statistics.fmean(mean for mean, _ in stats.values()) if stats else None,
            "noise": statistics.fmean(std for _, std in stats.values()) if stats else None,
        }
    return summary


def compare_checks(before: dict[str, Any], after: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark comparison: {before['run']} → {after['run']}",
        "",
        f"Before: git {before['manifest']['git_sha'][:10]}{' (dirty)' if before['manifest']['git_dirty'] else ''}, "
        f"{before['manifest']['settings']['provider']}/{before['manifest']['settings']['model']}",
        f"After:  git {after['manifest']['git_sha'][:10]}{' (dirty)' if after['manifest']['git_dirty'] else ''}, "
        f"{after['manifest']['settings']['provider']}/{after['manifest']['settings']['model']}",
        "",
        "A change counts only when it is larger than the run-to-run noise (mean per-paper spread across repeats) "
        "and moves the same way on most papers.",
        "",
        "| Metric | Before | After | Papers better / worse | Verdict |",
        "|---|---|---|---|---|",
    ]
    for label, key, direction, style in METRICS:
        stats_before, stats_after = per_paper_stats(before, key), per_paper_stats(after, key)
        shared = sorted(set(stats_before) & set(stats_after))
        if not shared:
            lines.append(f"| {label} | - | - | - | no data |")
            continue
        mean_before = statistics.fmean(stats_before[paper][0] for paper in shared)
        mean_after = statistics.fmean(stats_after[paper][0] for paper in shared)
        noise = max(statistics.fmean(stats_before[paper][1] for paper in shared),
                    statistics.fmean(stats_after[paper][1] for paper in shared))
        if direction == "info":
            lines.append(f"| {label} | {_format(mean_before, style)} | {_format(mean_after, style)} | - | info |")
            continue
        sign = 1 if direction == "higher" else -1
        better = sum(1 for paper in shared if sign * (stats_after[paper][0] - stats_before[paper][0]) > 1e-9)
        worse = sum(1 for paper in shared if sign * (stats_after[paper][0] - stats_before[paper][0]) < -1e-9)
        delta = sign * (mean_after - mean_before)
        if abs(mean_after - mean_before) <= noise or abs(mean_after - mean_before) < 1e-9:
            verdict = "≈ within noise"
        elif delta > 0 and better > worse:
            verdict = "✅ better"
        elif delta < 0 and worse > better:
            verdict = "⚠️ worse"
        else:
            verdict = "mixed"
        lines.append(f"| {label} | {_format(mean_before, style)} | {_format(mean_after, style)} | {better} / {worse} | {verdict} |")

    def issue_counts(checks: dict[str, Any]) -> dict[tuple[str, str], float]:
        repeats = max(1, checks["manifest"]["repeats"])
        counts: dict[tuple[str, str], float] = {}
        for run in checks["runs"]:
            for issue in run["issues"]:
                key = (issue["paper_id"], issue["category"])
                counts[key] = counts.get(key, 0) + 1 / repeats
        return counts

    counts_before, counts_after = issue_counts(before), issue_counts(after)
    regressions = sorted(
        ((key, counts_after[key] - counts_before.get(key, 0.0)) for key in counts_after
         if counts_after[key] - counts_before.get(key, 0.0) >= 1),
        key=lambda item: -item[1],
    )
    lines += ["", "## Papers where an issue type became more frequent", ""]
    lines += [f"- {paper}: {category} (+{increase:.1f} per run)" for (paper, category), increase in regressions] or ["- none"]
    return "\n".join(lines) + "\n"


def report_checks(checks: dict[str, Any], top: int = 8) -> str:
    manifest = checks["manifest"]
    lines = [
        f"# Benchmark report: {checks['run']}",
        "",
        f"git {manifest['git_sha'][:10]}{' (dirty)' if manifest['git_dirty'] else ''}, "
        f"{manifest['settings']['provider']}/{manifest['settings']['model']}, {len(manifest['paper_ids'])} papers × {manifest['repeats']} repeats",
        "",
        "| Metric | Mean | Noise (± across repeats) |",
        "|---|---|---|",
    ]
    summary = summarize(checks)
    for label, key, _direction, style in METRICS:
        values = summary[key]
        lines.append(f"| {label} | {_format(values['mean'], style)} | {_format(values['noise'], style)} |")
    by_category: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in checks["runs"]:
        for issue in run["issues"]:
            by_category.setdefault((issue["category"], issue["component"]), []).append(issue)
    lines += ["", "## Most frequent issues", "", "| Issue | Likely component | Count | Papers |", "|---|---|---|---|"]
    ranked = sorted(by_category.items(), key=lambda item: -len(item[1]))
    for (category, component), issues in ranked[:top]:
        lines.append(f"| {category} | {component} | {len(issues)} | {len({issue['paper_id'] for issue in issues})} |")
    lines += ["", "Full issue records with text and references are in checks.json."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepDoc benchmark runner.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download the PDFs of a paper set and verify their checksums.")
    fetch.add_argument("--set", default="dev")
    run = commands.add_parser("run", help="Run the production pipeline on a paper set and save every output.")
    run.add_argument("--set", default="dev")
    run.add_argument("--provider", required=True, choices=["ollama", "gemini"])
    run.add_argument("--model", required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--tag", required=True, help="Name for this run, e.g. v1-baseline.")
    run.add_argument("--summary-mode", choices=["paragraph", "standard", "one_page"])
    check = commands.add_parser("check", help="Run the automatic checks on a saved run and print a report.")
    check.add_argument("run", help="Tag, or tag/config when a tag holds several configurations.")
    compare = commands.add_parser("compare", help="Compare two saved runs.")
    compare.add_argument("before")
    compare.add_argument("after")
    args = parser.parse_args(argv)

    try:
        if args.command == "fetch":
            problems = fetch_papers(load_paper_set(args.set))
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            print("All PDFs present and verified." if not problems else f"{len(problems)} problem(s).")
            return 1 if problems else 0
        if args.command == "run":
            paper_set = load_paper_set(args.set)
            output_dir = run_benchmark(paper_set, args.tag, args.provider, args.model, args.repeats, args.summary_mode)
            checks = check_benchmark(output_dir, paper_set=paper_set)
            print(report_checks(checks))
            return 0
        if args.command == "check":
            output_dir = resolve_run_dir(args.run)
            report = report_checks(check_benchmark(output_dir))
            (output_dir / "report.md").write_text(report, encoding="utf-8")
            print(report)
            return 0
        if args.command == "compare":
            dirs = [resolve_run_dir(spec) for spec in (args.before, args.after)]
            checks = [check_benchmark(directory) for directory in dirs]
            report = compare_checks(*checks)
            target = RUN_DIR / f"compare-{args.before.replace('/', '_')}-vs-{args.after.replace('/', '_')}.md"
            target.write_text(report, encoding="utf-8")
            print(report)
            print(f"Saved to {target}")
            return 0
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
