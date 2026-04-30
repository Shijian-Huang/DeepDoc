from typing import cast

import fitz  # PyMuPDF

def parse_pdf(path: str) -> str:
    texts: list[str] = []

    with fitz.open(path) as doc:
        for page in doc:
            text = cast(str, page.get_text("text"))
            if text.strip():
                texts.append(text)
    
    return "\n".join(texts)
