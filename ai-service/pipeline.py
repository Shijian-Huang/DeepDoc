import re

from llm.summarizer import summarize_research_paper
from parser.pdf_parser import parse_pdf
from parser.reference_extractor import extract_references
from utils.section_extractor import build_summary_input

REFERENCE_HEADING_RE = re.compile(r"(?im)^\s*(references|bibliography)\s*$")

def remove_references_section(text: str) -> str:
    match = REFERENCE_HEADING_RE.search(text)
    if match:
        return text[:match.start()]

    return text

def run_pipeline(file_path: str):
    # Parse PDF
    text = parse_pdf(file_path)
    if not text.strip():
        return {
            "document_summary": {
                "summary": "No readable text was found in the PDF.",
                "key_ideas": [],
                "contributions": []
            },
            "chunk_summaries": [],
            "references": []
        }

    body_text = remove_references_section(text)
    summary_input, selected_sections = build_summary_input(body_text)

    if len(summary_input.strip()) < 200:
        return {
            "document_summary": {
                "summary": "The PDF text was extracted, but no chunk was long enough to summarize.",
                "key_ideas": [],
                "contributions": []
            },
            "chunk_summaries": [],
            "references": extract_references(text)
        }

    final_summary = summarize_research_paper(summary_input)
    references = extract_references(text)

    return {
        "document_summary": final_summary,
        # "chunk_summaries": [],
        "references": references,
        "summary_input_sections": selected_sections
    }
