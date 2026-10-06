import time

from llm.summarizer import normalize_summary_mode, summarize_research_paper
from parser.document_parser import extract_document_title, parse_document_pages
from parser.reference_extractor import extract_references_with_diagnostics
from utils.section_extractor import build_summary_input_from_pages

def run_pipeline(file_path: str, summary_mode: str = "standard"):
    normalized_mode = normalize_summary_mode(summary_mode)
    started = time.perf_counter()
    diagnostics = {"stage_seconds": {}}

    pages = parse_document_pages(file_path)
    paper_title = extract_document_title(file_path, pages)
    diagnostics["stage_seconds"]["parse"] = round(time.perf_counter() - started, 4)
    text = "\n".join(page["text"] for page in pages)
    raw_text = "\n".join(page.get("raw_text", page["text"]) for page in pages)
    if not text.strip():
        return {
            "pipeline_diagnostics": {**diagnostics, "status": "no_readable_text"},
            "paper_title": paper_title,
            "summary_mode": normalized_mode,
            "document_summary": {
                "title": paper_title,
                "summary": "No readable text was found in the document.",
                "key_ideas": [],
                "contributions": [],
                "evidence": []
            },
            "chunk_summaries": [],
            "references": [],
            "evidence_sources": [],
            "page_count": len(pages)
        }

    selection_started = time.perf_counter()
    summary_input, selected_sections, evidence_sources = build_summary_input_from_pages(
        pages,
        summary_mode=normalized_mode,
    )
    diagnostics["stage_seconds"]["section_selection"] = round(time.perf_counter() - selection_started, 4)
    diagnostics["evidence_source_count"] = len(evidence_sources)
    diagnostics["summary_input_chars"] = len(summary_input)

    if len(summary_input.strip()) < 200:
        return {
            "pipeline_diagnostics": {**diagnostics, "status": "insufficient_text"},
            "paper_title": paper_title,
            "summary_mode": normalized_mode,
            "document_summary": {
                "title": paper_title,
                "summary": "The document text was extracted, but no chunk was long enough to summarize.",
                "key_ideas": [],
                "contributions": [],
                "evidence": []
            },
            "chunk_summaries": [],
            **_reference_result_fields(raw_text),
            "summary_input_sections": selected_sections,
            "evidence_sources": evidence_sources,
            "page_count": len(pages)
        }

    summary_started = time.perf_counter()
    final_summary = summarize_research_paper(
        summary_input,
        summary_mode=normalized_mode,
        evidence_sources=evidence_sources,
    )
    diagnostics["stage_seconds"]["summary"] = round(time.perf_counter() - summary_started, 4)
    if final_summary.get("error"):
        raise RuntimeError(str(final_summary["error"]))
    final_summary.setdefault("title", paper_title)
    references_started = time.perf_counter()
    reference_fields = _reference_result_fields(raw_text)
    diagnostics["stage_seconds"]["references"] = round(time.perf_counter() - references_started, 4)
    diagnostics["total_seconds"] = round(time.perf_counter() - started, 4)
    diagnostics["status"] = "completed"

    return {
        "pipeline_diagnostics": diagnostics,
        "paper_title": paper_title,
        "summary_mode": normalized_mode,
        "document_summary": final_summary,
        # "chunk_summaries": [],
        **reference_fields,
        "summary_input_sections": selected_sections,
        "evidence_sources": evidence_sources,
        "page_count": len(pages)
    }


def _reference_result_fields(raw_text: str) -> dict:
    result = extract_references_with_diagnostics(raw_text)
    fields = {
        "references": result.references,
        "references_extraction_method": result.method,
        "references_low_confidence": result.low_confidence,
        "references_expected_count": result.expected_count,
        "references_section_char_count": result.section_char_count,
        "references_extraction_notes": result.notes,
    }
    if result.repaired:
        fields["references_repaired_from_pdf"] = True
    return fields
