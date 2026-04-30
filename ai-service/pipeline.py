from llm.summarizer import summarize_chunk, summarize_document
from parser.pdf_parser import parse_pdf
from utils.chunker import chunk_text

def run_pipeline(file_path: str):
    # Parse PDF
    text = parse_pdf(file_path)

    # Chunking
    chunks = chunk_text(text)

    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        result = summarize_chunk(chunk)
        chunk_summaries.append(result)

    final_summary = summarize_document(chunk_summaries)

    return {
        "document_summary": final_summary,
        "chunk_summaries": chunk_summaries
    }
