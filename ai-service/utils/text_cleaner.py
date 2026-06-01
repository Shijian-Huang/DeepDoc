import re
from collections import Counter
from typing import Optional


HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVXLCM]+\.?\s+|[A-Z]\.\s+)?"
)

SECTION_HEADING_ALIASES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "prior work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "system",
    "architecture",
    "experiment",
    "experiments",
    "evaluation",
    "setup",
    "result",
    "results",
    "discussion",
    "analysis",
    "findings",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "bibliography",
}

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

NOISE_LINE_RE = re.compile(
    r"(?i)("
    r"^©\s*\d{4}|"
    r"^copyright\b|"
    r"^acm isbn\b|"
    r"^https?://doi\.org/|"
    r"^doi:\s*|"
    r"^permission to make digital or hard copies|"
    r"^to copy otherwise, or republish|"
    r"^request permissions from permissions@|"
    r"^proceedings of the|"
    r"^published by the association for computing machinery"
    r")"
)

VENUE_HEADER_RE = re.compile(
    r"(?i)^[A-Z][A-Z0-9/ .'-]{2,20}\s*[’']?\d{2},\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\b"
)

COMPOUND_SUFFIXES = {
    "aware",
    "based",
    "centric",
    "driven",
    "end",
    "free",
    "level",
    "like",
    "oriented",
    "specific",
}


def normalize_line_for_frequency(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def find_repeated_noise_lines(raw_pages: list[str]) -> set[str]:
    counts: Counter[str] = Counter()

    for page_text in raw_pages:
        seen_on_page: set[str] = set()
        for line in page_text.splitlines():
            normalized = normalize_line_for_frequency(line)
            if 4 <= len(normalized) <= 140:
                seen_on_page.add(normalized)

        counts.update(seen_on_page)

    min_repetitions = 2 if len(raw_pages) <= 6 else 3
    return {
        line
        for line, count in counts.items()
        if count >= min_repetitions and _looks_like_repeated_noise(line)
    }


def _looks_like_repeated_noise(normalized_line: str) -> bool:
    return (
        "copyright" in normalized_line
        or "isbn" in normalized_line
        or "doi.org" in normalized_line
        or "proceedings" in normalized_line
        or re.search(r"\b[a-z]+ ['’]\d{2},", normalized_line) is not None
    )


def should_drop_line(line: str, repeated_noise_lines: set[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    normalized = normalize_line_for_frequency(stripped)
    return (
        PAGE_NUMBER_RE.match(stripped) is not None
        or NOISE_LINE_RE.search(stripped) is not None
        or VENUE_HEADER_RE.match(stripped) is not None
        or normalized in repeated_noise_lines
    )


def is_heading_like(line: str) -> bool:
    stripped = line.strip()
    cleaned = HEADING_PREFIX_RE.sub("", stripped).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in SECTION_HEADING_ALIASES:
        return True

    return (
        len(stripped) <= 90
        and stripped.upper() == stripped
        and re.search(r"[A-Z]", stripped) is not None
    )


def _join_wrapped_lines(lines: list[str]) -> str:
    output: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return

        output.append(re.sub(r"\s+", " ", " ".join(paragraph)).strip())
        paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        if is_heading_like(stripped):
            flush_paragraph()
            output.append(stripped)
            continue

        paragraph.append(stripped)

    flush_paragraph()
    return "\n".join(output)


def _repair_hyphenated_line_breaks(text: str) -> str:
    def replace(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if right.lower() in COMPOUND_SUFFIXES:
            return f"{left}-{right}"

        return f"{left}{right}"

    return re.sub(r"(?<=\b)(\w+)-\s*\n\s*(\w+)(?=\b)", replace, text)


def clean_pdf_page_text(text: str, repeated_noise_lines: Optional[set[str]] = None) -> str:
    repeated_noise_lines = repeated_noise_lines or set()
    text = text.replace("\x00", " ")
    text = _repair_hyphenated_line_breaks(text)

    kept_lines = [
        line
        for line in text.splitlines()
        if not should_drop_line(line, repeated_noise_lines)
    ]

    cleaned = _join_wrapped_lines(kept_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
