import re
from typing import cast

import fitz  # PyMuPDF

from utils.text_cleaner import clean_pdf_page_text, find_repeated_noise_lines


TITLE_STOP_LINES = {
    "abstract",
    "introduction",
    "keywords",
    "acm reference format:",
    "ccs concepts",
}


ARTICLE_TYPE_LINES = {
    "article",
    "research article",
    "review article",
    "original article",
    "survey article",
}


def _clean_title_candidate(value: str) -> str:
    return " ".join(value.replace("\x00", " ").split()).strip(" -")


def _looks_like_publication_header(value: str) -> bool:
    cleaned = _clean_title_candidate(value)
    lowered = cleaned.lower()
    if not cleaned:
        return True
    if lowered in ARTICLE_TYPE_LINES:
        return True
    if any(token in lowered for token in [
        "contents lists available",
        "journal homepage",
        "sciencedirect",
        "elsevier",
        "springer",
        "proceedings of",
    ]):
        return True
    if re.search(r"\b\d+\s*\(\d{4}\)\s*\d+\b", cleaned):
        return True
    return False


def _looks_like_metadata_title(value: str) -> bool:
    cleaned = _clean_title_candidate(value)
    if len(cleaned) < 8:
        return False
    lowered = cleaned.lower()
    return (
        not any(token in lowered for token in ["untitled", "microsoft word", ".pdf"])
        and not _looks_like_publication_header(cleaned)
    )


def _looks_like_author_or_affiliation(value: str) -> bool:
    lowered = value.lower()
    return (
        "@" in value
        or lowered in {"u.s.a.", "usa"}
        or bool(re.search(r"\buniversity\b|\binstitute\b|\bcollege\b|\bdepartment\b|\bgoogle\b|\binc\b", lowered))
        or "∗" in value
    )


def extract_pdf_title(path: str, pages: list[dict] | None = None) -> str:
    try:
        with fitz.open(path) as doc:
            metadata_title = (doc.metadata or {}).get("title") or ""
            if _looks_like_metadata_title(metadata_title):
                return _clean_title_candidate(metadata_title)
    except Exception:
        pass

    source_pages = pages or parse_pdf_pages(path)
    if not source_pages:
        return ""

    first_page_text = source_pages[0].get("raw_text") or source_pages[0].get("text") or ""
    lines = [_clean_title_candidate(line) for line in first_page_text.splitlines()]
    lines = [line for line in lines if line]

    title_lines: list[str] = []
    saw_article_type = False
    candidate_lines = lines[:30]
    for offset, line in enumerate(candidate_lines):
        lowered = line.lower()
        if lowered in TITLE_STOP_LINES or lowered.startswith("abstract"):
            break
        if lowered in ARTICLE_TYPE_LINES:
            saw_article_type = True
            title_lines = []
            continue
        next_line = candidate_lines[offset + 1].lower() if offset + 1 < len(candidate_lines) else ""
        if "journal homepage" in next_line and not title_lines:
            continue
        if _looks_like_publication_header(line) and not title_lines:
            continue
        if _looks_like_author_or_affiliation(line):
            break
        title_lines.append(line)
        if saw_article_type and len(title_lines) >= 2:
            break
        if not saw_article_type and len(" ".join(title_lines)) >= 16:
            break

    return _clean_title_candidate(" ".join(title_lines))


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
