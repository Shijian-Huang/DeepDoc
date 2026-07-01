#!/usr/bin/env python3
import argparse
import json
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parser.pdf_parser import extract_pdf_title, parse_pdf_pages
from parser.reference_extractor import extract_references
from pipeline import run_pipeline
from services.arxiv_service import ArxivServiceError, download_arxiv_pdf, search_arxiv
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


def sample_arxiv_papers(categories: list[str], count: int, max_start: int, seed: int | None) -> list[dict]:
    rng = random.Random(seed)
    papers: list[dict] = []
    seen_ids: set[str] = set()
    attempts = 0

    while len(papers) < count and attempts < count * 8:
        attempts += 1
        category = rng.choice(categories)
        start = rng.randint(0, max_start)
        query = f"cat:{category}"
        try:
            results = search_arxiv_with_start(query, max_results=5, start=start)
        except ArxivServiceError as error:
            print(f"[warn] arXiv search failed for {query} start={start}: {error}", file=sys.stderr)
            time.sleep(1.0)
            continue

        rng.shuffle(results)
        for paper in results:
            arxiv_id = paper.get("arxiv_id")
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            papers.append(paper)
            break

        time.sleep(0.35)

    return papers[:count]


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
    except Exception as error:
        raise ArxivServiceError("Could not search arXiv for regression sampling.") from error

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
        result["pipeline"] = {
            "summary_mode": pipeline_result.get("summary_mode"),
            "summary_word_count": summary.get("summary_word_count"),
            "evidence_count": len(summary.get("evidence") or []),
            "reference_count": len(pipeline_result.get("references") or []),
            "summary_input_sections": pipeline_result.get("summary_input_sections") or [],
        }
        if not summary.get("evidence"):
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
            print(f"[{index}/{len(args.pdfs)}] local PDF {pdf_path.name}")
            item = {
                "arxiv_id": pdf_path.stem,
                "title": pdf_path.stem,
                "pdf_url": str(pdf_path),
                "categories": ["local"],
            }
            try:
                item["diagnostics"] = diagnose_pdf(pdf_path, args.summary_mode, args.run_llm)
            except Exception as error:
                item["diagnostics"] = {"flags": ["eval_failed"], "error": str(error)}
            results.append(item)
    else:
        papers = sample_arxiv_papers(args.categories, args.count, args.max_start, args.seed)
        for index, paper in enumerate(papers, start=1):
            arxiv_id = paper.get("arxiv_id", "unknown")
            print(f"[{index}/{len(papers)}] {arxiv_id} {paper.get('title', '')[:80]}")
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
