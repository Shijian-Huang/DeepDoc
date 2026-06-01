from typing import cast

import fitz  # PyMuPDF

from utils.text_cleaner import clean_pdf_page_text, find_repeated_noise_lines


def parse_pdf_pages(path: str) -> list[dict]:
    pages: list[dict] = []

    with fitz.open(path) as doc:
        raw_pages = [
            cast(str, page.get_text("text"))
            for page in doc
        ]

        repeated_noise_lines = find_repeated_noise_lines(raw_pages)

        for index, raw_text in enumerate(raw_pages):
            text = clean_pdf_page_text(raw_text, repeated_noise_lines)
            if text.strip():
                pages.append({
                    "page": index + 1,
                    "raw_text": raw_text,
                    "text": text,
                })

    return pages


def parse_pdf(path: str) -> str:
    texts = [page["text"] for page in parse_pdf_pages(path)]
    
    return "\n".join(texts)
