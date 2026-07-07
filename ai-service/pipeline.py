from llm.summarizer import normalize_summary_mode, summarize_research_paper
from parser.pdf_parser import extract_pdf_title, parse_pdf_pages
from parser.reference_extractor import extract_references
from utils.section_extractor import build_summary_input_from_pages

def run_pipeline(file_path: str, summary_mode: str = "standard"):
    normalized_mode = normalize_summary_mode(summary_mode)

    # Parse PDF
    pages = parse_pdf_pages(file_path)
    paper_title = extract_pdf_title(file_path, pages)
    text = "\n".join(page["text"] for page in pages)
    raw_text = "\n".join(page.get("raw_text", page["text"]) for page in pages)
    if not text.strip():
        return {
            "paper_title": paper_title,
            "summary_mode": normalized_mode,
            "document_summary": {
                "title": paper_title,
                "summary": "No readable text was found in the PDF.",
                "key_ideas": [],
                "contributions": [],
                "evidence": []
            },
            "chunk_summaries": [],
            "references": [],
            "evidence_sources": [],
            "page_count": len(pages)
        }

    summary_input, selected_sections, evidence_sources = build_summary_input_from_pages(
        pages,
        summary_mode=normalized_mode,
    )

    if len(summary_input.strip()) < 200:
        return {
            "paper_title": paper_title,
            "summary_mode": normalized_mode,
            "document_summary": {
                "title": paper_title,
                "summary": "The PDF text was extracted, but no chunk was long enough to summarize.",
                "key_ideas": [],
                "contributions": [],
                "evidence": []
            },
            "chunk_summaries": [],
            "references": extract_references(raw_text),
            "summary_input_sections": selected_sections,
            "evidence_sources": evidence_sources,
            "page_count": len(pages)
        }

    final_summary = summarize_research_paper(
        summary_input,
        summary_mode=normalized_mode,
    )
    final_summary.setdefault("title", paper_title)
    references = extract_references(raw_text)

    return {
        "paper_title": paper_title,
        "summary_mode": normalized_mode,
        "document_summary": final_summary,
        # "chunk_summaries": [],
        "references": references,
        "summary_input_sections": selected_sections,
        "evidence_sources": evidence_sources,
        "page_count": len(pages)
    }
