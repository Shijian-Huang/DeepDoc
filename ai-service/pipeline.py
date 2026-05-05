from llm.summarizer import summarize_chunk, summarize_document
from parser.pdf_parser import parse_pdf
from parser.reference_extractor import extract_references
from utils.chunker import chunk_text

MAX_SUMMARY_CHUNKS = 4
REFERENCE_KEYWORDS = ["References", "REFERENCES", "bibliography", "Bibliography"]

def remove_references_section(text: str) -> str:
    for keyword in REFERENCE_KEYWORDS:
        if keyword in text:
            return text.split(keyword)[0]

    return text

def sample_chunks(chunks: list[str], k: int = MAX_SUMMARY_CHUNKS) -> list[str]:
    if len(chunks) <= k:
        return chunks

    if k == 1:
        return [chunks[0]]

    last_index = len(chunks) - 1
    indices = [round(i * last_index / (k - 1)) for i in range(k)]
    return [chunks[i] for i in indices]

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

    # Chunking
    body_text = remove_references_section(text)
    chunks = sample_chunks(chunk_text(body_text))

    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        if len(chunk.strip()) < 200:
            continue
        try:
            result = summarize_chunk(chunk)
        except Exception as error:
            result = {
                "summary": f"Chunk summarization failed: {error}",
                "key_points": []
            }
        chunk_summaries.append(result)

    successful_chunk_summaries = [
        summary for summary in chunk_summaries
        if summary.get("summary")
    ]

    if not successful_chunk_summaries:
        return {
            "document_summary": {
                "summary": "The PDF text was extracted, but no chunk was long enough to summarize.",
                "key_ideas": [],
                "contributions": []
            },
            "chunk_summaries": [],
            "references": extract_references(text)
        }

    final_summary = summarize_document(successful_chunk_summaries)

    references = extract_references(text)

    return {
        "document_summary": final_summary,
        "chunk_summaries": chunk_summaries,
        "references": references
    }
