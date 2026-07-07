#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parser.pdf_parser import extract_pdf_title, parse_pdf_pages
from parser.reference_extractor import extract_references
from pipeline import run_pipeline
from services.arxiv_service import ArxivServiceError, download_arxiv_pdf
from utils.section_extractor import build_summary_input_from_pages, split_pages_into_sections


DEFAULT_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.HC",
    "cs.LG",
    "cs.SE",
    "stat.ML",
]

REPORT_DIR = ROOT / "data" / "eval_reports"
PDF_DIR = ROOT / "data" / "eval_pdfs"

SUMMARY_MIN_WORDS = {
    "paragraph": 70,
    "standard": 170,
    "one_page": 300,
}


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text or "")))


def text_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "paper",
        "study", "method", "approach", "results", "show", "shows", "using",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", str(text or "").lower())
        if token not in stopwords
    }


def has_near_duplicate(left_items: list, right_items: list) -> bool:
    for left in left_items:
        left_tokens = text_tokens(left)
        if not left_tokens:
            continue
        for right in right_items:
            right_tokens = text_tokens(right)
            if not right_tokens:
                continue
            overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
            if overlap >= 0.78:
                return True
    return False


def sample_arxiv_papers(categories: list[str], count: int, max_start: int, seed: int | None) -> list[dict]:
    rng = random.Random(seed)
    papers: list[dict] = []
    seen_ids: set[str] = set()
    attempts = 0
    failure_streak = 0

    while len(papers) < count and attempts < count * 6:
        attempts += 1
        category = rng.choice(categories)
        start = rng.randint(0, max_start)
        query = f"cat:{category}"
        try:
            results = search_arxiv_with_start(query, max_results=5, start=start)
        except ArxivServiceError as error:
            failure_streak += 1
            if failure_streak <= 3 or failure_streak % 10 == 0:
                print(f"[warn] arXiv search failed for {query} start={start}: {error}", file=sys.stderr)
            time.sleep(min(12.0, 1.5 * failure_streak))
            continue

        failure_streak = 0
        rng.shuffle(results)
        for paper in results:
            arxiv_id = paper.get("arxiv_id")
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            papers.append(paper)
            break

        time.sleep(1.1)

    if len(papers) < count:
        print(
            f"[warn] Random arXiv sampling found {len(papers)}/{count}; trying latest-paper fallback.",
            file=sys.stderr,
        )
        for paper in latest_arxiv_papers(categories, count * 2):
            arxiv_id = paper.get("arxiv_id")
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            papers.append(paper)
            if len(papers) >= count:
                break

    return papers[:count]


def latest_arxiv_papers(categories: list[str], limit: int) -> list[dict]:
    papers: list[dict] = []
    for category in categories:
        if len(papers) >= limit:
            break
        query = f"cat:{category}"
        try:
            papers.extend(search_arxiv_with_start(query, max_results=min(25, limit - len(papers)), start=0))
        except ArxivServiceError as error:
            print(f"[warn] Latest-paper fallback failed for {query}: {error}", file=sys.stderr)
            time.sleep(8.0)
        time.sleep(3.0)
    return papers[:limit]


def search_arxiv_with_start(query: str, max_results: int, start: int) -> list[dict]:
    # arxiv_service.search_arxiv intentionally exposes the app route behavior.
    # This helper keeps regression sampling random without changing production search.
    from services import arxiv_service
    from urllib.request import Request, urlopen
    from xml.etree import ElementTree

    params = urlencode({
        "search_query": query,
        "start": max(0, int(start)),
        "max_results": max(1, min(int(max_results), 25)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    request = Request(
        f"{arxiv_service.ARXIV_API_URL}?{params}",
        headers={"User-Agent": "DeepDoc/0.1 (arXiv regression evaluator)"},
    )
    try:
        with urlopen(request, timeout=arxiv_service.DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except HTTPError as error:
        if error.code == 429:
            raise ArxivServiceError("arXiv rate limit reached; retry after a short pause.") from error
        raise ArxivServiceError(f"arXiv search failed with HTTP {error.code}.") from error
    except (URLError, TimeoutError) as error:
        raise ArxivServiceError("Could not reach arXiv for regression sampling.") from error

    root = ElementTree.fromstring(payload)
    papers: list[dict] = []
    for entry in root.findall("atom:entry", arxiv_service.ATOM_NS):
        abs_url = arxiv_service._text(entry.find("atom:id", arxiv_service.ATOM_NS))
        arxiv_id = arxiv_service._arxiv_id_from_abs_url(abs_url)
        authors = [
            arxiv_service._text(author.find("atom:name", arxiv_service.ATOM_NS))
            for author in entry.findall("atom:author", arxiv_service.ATOM_NS)
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", arxiv_service.ATOM_NS)
            if category.attrib.get("term")
        ]
        papers.append({
            "title": arxiv_service._text(entry.find("atom:title", arxiv_service.ATOM_NS)),
            "authors": [author for author in authors if author],
            "summary": arxiv_service._text(entry.find("atom:summary", arxiv_service.ATOM_NS)),
            "published": arxiv_service._text(entry.find("atom:published", arxiv_service.ATOM_NS))[:10],
            "updated": arxiv_service._text(entry.find("atom:updated", arxiv_service.ATOM_NS))[:10],
            "arxiv_id": arxiv_id,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": arxiv_service._pdf_url_from_entry(entry, arxiv_id),
            "categories": categories,
        })
    return papers


def diagnose_pdf(pdf_path: Path, summary_mode: str, run_llm: bool) -> dict:
    pages = parse_pdf_pages(str(pdf_path))
    full_text = "\n".join(page.get("raw_text", page["text"]) for page in pages)
    sections = split_pages_into_sections(pages)
    summary_packet, selected_sections, evidence_sources = build_summary_input_from_pages(
        pages,
        summary_mode=summary_mode,
    )
    references = extract_references(full_text)
    title = extract_pdf_title(str(pdf_path), pages)

    result: dict = {
        "pdf_filename": pdf_path.name,
        "title": title,
        "page_count": len(pages),
        "text_chars": sum(len(page.get("text", "")) for page in pages),
        "sections": {
            name: {
                "pages": value.get("pages", []),
                "chars": len(value.get("text", "")),
            }
            for name, value in sections.items()
        },
        "selected_sections": selected_sections,
        "evidence_source_count": len(evidence_sources),
        "reference_count": len(references),
        "summary_packet_chars": len(summary_packet),
        "flags": [],
    }

    flags = result["flags"]
    if result["page_count"] == 0 or result["text_chars"] < 1000:
        flags.append("low_text_extraction")
    if len(selected_sections) <= 1:
        flags.append("single_section_summary_input")
    if not {"introduction", "abstract", "preamble"} & set(sections):
        flags.append("missing_intro_or_preamble")
    if "references" not in sections and result["reference_count"] == 0:
        flags.append("missing_references")
    if result["reference_count"] and result["reference_count"] < 5:
        flags.append("low_reference_count")
    if not result["title"] or len(result["title"]) < 8:
        flags.append("weak_title")

    if run_llm:
        pipeline_result = run_pipeline(str(pdf_path), summary_mode=summary_mode)
        summary = pipeline_result.get("document_summary", {})
        key_ideas = summary.get("key_ideas") if isinstance(summary.get("key_ideas"), list) else []
        contributions = summary.get("contributions") if isinstance(summary.get("contributions"), list) else []
        evidence = summary.get("evidence") if isinstance(summary.get("evidence"), list) else []
        limitations = summary.get("limitations") if isinstance(summary.get("limitations"), list) else []
        discussion_questions = (
            summary.get("discussion_questions")
            if isinstance(summary.get("discussion_questions"), list)
            else []
        )
        summary_text = str(summary.get("summary") or "")
        summary_words = int(summary.get("summary_word_count") or count_words(summary_text))
        result["pipeline"] = {
            "summary_mode": pipeline_result.get("summary_mode"),
            "summary_word_count": summary_words,
            "key_idea_count": len(key_ideas),
            "contribution_count": len(contributions),
            "limitation_count": len(limitations),
            "discussion_question_count": len(discussion_questions),
            "evidence_count": len(evidence),
            "reference_count": len(pipeline_result.get("references") or []),
            "summary_input_sections": pipeline_result.get("summary_input_sections") or [],
        }
        if summary_words < SUMMARY_MIN_WORDS.get(summary_mode, 170):
            flags.append("pipeline_short_summary")
        if len(key_ideas) < 3:
            flags.append("pipeline_few_key_ideas")
        if len(contributions) < 2:
            flags.append("pipeline_few_contributions")
        if has_near_duplicate(key_ideas, contributions):
            flags.append("pipeline_key_contribution_overlap")
        if len(evidence) < 3:
            flags.append("pipeline_missing_evidence")
        if len(pipeline_result.get("references") or []) == 0:
            flags.append("pipeline_missing_references")

    return result


def write_reports(results: list[dict], output_prefix: Path) -> tuple[Path, Path]:
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    lines = [
        "# DeepDoc arXiv Regression Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Papers: {len(results)}",
        "",
        "## Summary",
        "",
    ]
    flagged = [item for item in results if item.get("diagnostics", {}).get("flags")]
    lines.append(f"- Flagged papers: {len(flagged)}")
    lines.append("")

    for item in results:
        diag = item.get("diagnostics", {})
        flags = diag.get("flags", [])
        lines.extend([
            f"## {item.get('arxiv_id', 'unknown')} - {item.get('title', 'Untitled')}",
            "",
            f"- Categories: {', '.join(item.get('categories') or []) or 'unknown'}",
            f"- PDF: {item.get('pdf_url', '')}",
            f"- Pages: {diag.get('page_count', 0)}",
            f"- Selected sections: {', '.join(diag.get('selected_sections') or []) or 'none'}",
            f"- References: {diag.get('reference_count', 0)}",
            f"- Evidence sources: {diag.get('evidence_source_count', 0)}",
            f"- Flags: {', '.join(flags) if flags else 'none'}",
            "",
        ])
        if diag.get("pipeline"):
            pipeline = diag["pipeline"]
            lines.extend([
                f"  - Summary words: {pipeline.get('summary_word_count', 0)}",
                f"  - Key ideas / contributions / evidence: {pipeline.get('key_idea_count', 0)} / {pipeline.get('contribution_count', 0)} / {pipeline.get('evidence_count', 0)}",
                f"  - Limitations / discussion questions: {pipeline.get('limitation_count', 0)} / {pipeline.get('discussion_question_count', 0)}",
                "",
            ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Randomly sample arXiv papers and diagnose DeepDoc parsing quality.")
    parser.add_argument("--count", type=int, default=5, help="Number of papers to sample.")
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES, help="arXiv categories to sample.")
    parser.add_argument("--max-start", type=int, default=150, help="Maximum random arXiv start offset.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible samples.")
    parser.add_argument("--pdfs", nargs="*", default=None, help="Local PDF paths to diagnose instead of sampling arXiv.")
    parser.add_argument("--summary-mode", default="standard", choices=["paragraph", "standard", "one_page"])
    parser.add_argument("--run-llm", action="store_true", help="Run the full Gemini-backed pipeline. This may cost API tokens.")
    parser.add_argument("--keep-pdfs", action="store_true", help="Keep downloaded PDFs under data/eval_pdfs.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = PDF_DIR if args.keep_pdfs else Path(tempfile.mkdtemp(prefix="deepdoc-arxiv-eval-"))

    results: list[dict] = []
    if args.pdfs:
        for index, raw_path in enumerate(args.pdfs, start=1):
            pdf_path = Path(raw_path).expanduser().resolve()
            print(f"[{index}/{len(args.pdfs)}] local PDF {pdf_path.name}", flush=True)
            item = {
                "arxiv_id": pdf_path.stem,
                "title": pdf_path.stem,
                "pdf_url": str(pdf_path),
                "categories": ["local"],
            }
            try:
                item["diagnostics"] = diagnose_pdf(pdf_path, args.summary_mode, args.run_llm)
                item["title"] = item["diagnostics"].get("title") or item["title"]
            except Exception as error:
                item["diagnostics"] = {"flags": ["eval_failed"], "error": str(error)}
            results.append(item)
    else:
        papers = sample_arxiv_papers(args.categories, args.count, args.max_start, args.seed)
        for index, paper in enumerate(papers, start=1):
            arxiv_id = paper.get("arxiv_id", "unknown")
            print(f"[{index}/{len(papers)}] {arxiv_id} {paper.get('title', '')[:80]}", flush=True)
            item = dict(paper)
            try:
                pdf_path = download_arxiv_pdf(paper["pdf_url"], arxiv_id, temp_dir)
                item["diagnostics"] = diagnose_pdf(pdf_path, args.summary_mode, args.run_llm)
            except Exception as error:
                item["diagnostics"] = {"flags": ["eval_failed"], "error": str(error)}
            results.append(item)
            time.sleep(0.35)

    if not results:
        print("No papers were evaluated.", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_prefix = REPORT_DIR / f"arxiv-regression-{timestamp}"
    json_path, md_path = write_reports(results, output_prefix)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
