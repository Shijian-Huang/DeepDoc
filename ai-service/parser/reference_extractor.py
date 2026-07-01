import re
from typing import Optional

from llm.summarizer import extract_references_llm


REFERENCE_SECTION_RE = re.compile(
    r"(?ims)^\s*(references(?:\s+and\s+notes)?|bibliography|literature\s+cited|cited\s+references)\s*$\n(?P<body>.*)"
)

BRACKET_REFERENCE_RE = re.compile(r"(?m)^\s*\[(\d+)\][ \t]+")
NUMBERED_REFERENCE_RE = re.compile(r"(?m)^\s*(\d{1,3})[.)][ \t]*(?=[A-Z])")

REFERENCE_START_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d{1,3}\.|\d{1,3}\)|[A-Z][A-Za-z'-]+,\s+[A-Z])"
)

TRAILING_SECTION_RE = re.compile(
    r"(?im)^\s*(appendix|appendices|acknowledg(e)?ments?|code\s+availability|data\s+availability|supplementary\s+materials?)\s*$"
)

APPENDIX_HEADING_RE = re.compile(r"^(appendix|appendices)\b", re.IGNORECASE)
APPENDIX_TITLE_RE = re.compile(r"^[A-Z].{4,}$")
YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
APPENDIX_MARKER_RE = re.compile(r"^[A-Z]$")
TRAILING_AUTHOR_GIVEN_NAME_RE = re.compile(
    r"^(?P<body>.+?\b(?:19|20)\d{2}[a-z]?(?:[.)]|\b)[^.]*)\.\s+"
    r"(?P<name>[A-Z][a-z]{2,})(?:-[A-Z][A-Za-z]+)?$"
)
LEADING_AUTHOR_SURNAME_RE = re.compile(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'.’:-]+,\s+")
INLINE_BRACKET_REFERENCES_RE = re.compile(r"(?ms)^\s*(?P<body>\[1\][ \t]+.+)")
ACKNOWLEDGMENT_REFERENCES_RE = re.compile(
    r"(?ims)^\s*acknowledg(?:e)?ments?\s*$.*?(?P<body>^\s*1(?=[A-Z]).*)"
)
SUPERSCRIPT_NUMBERED_REFERENCE_RE = re.compile(r"(?m)^\s*(\d{1,3})(?=[A-Z])")


def extract_reference_section(text: str) -> str:
    match = REFERENCE_SECTION_RE.search(text)
    if match:
        ref_text = match.group("body")
    else:
        inline_match = INLINE_BRACKET_REFERENCES_RE.search(text)
        if inline_match:
            ref_text = inline_match.group("body")
        else:
            acknowledgment_match = ACKNOWLEDGMENT_REFERENCES_RE.search(text)
            if not acknowledgment_match:
                return ""
            ref_text = acknowledgment_match.group("body")

    trailing_match = TRAILING_SECTION_RE.search(ref_text)
    if trailing_match:
        ref_text = ref_text[:trailing_match.start()]

    raw_lines = ref_text.splitlines()
    lines: list[str] = []
    for index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if lines and APPENDIX_HEADING_RE.match(line):
            break
        next_line = ""
        for candidate in raw_lines[index + 1:]:
            if candidate.strip():
                next_line = candidate.strip()
                break
        if lines and APPENDIX_MARKER_RE.match(line) and APPENDIX_TITLE_RE.match(next_line):
            break
        lines.append(raw_line)

    ref_text = "\n".join(lines)
    return ref_text.strip()


def _clean_reference_entry(entry: str) -> str:
    entry = re.sub(r"\s+", " ", entry).strip()
    entry = re.sub(r"(?<=\w)-\s+(?=\w)", "", entry)
    entry = re.sub(r"https:\s+//", "https://", entry)
    entry = re.sub(r"/\s+", "/", entry)
    entry = re.sub(r"\s+(?:FIG\.|Figure|TABLE|Table)\s+\d+\..*$", "", entry)
    entry = re.sub(r"(?<=\(\d{4}\)\.)\s+[A-Z][^.]{20,}\s+\d{1,3}$", "", entry)
    entry = re.sub(r"(?<=\.)\s+\d{1,3}$", "", entry)
    return entry


def _repair_reference_boundaries(entries: list[str]) -> list[str]:
    repaired: list[str] = []
    index = 0
    while index < len(entries):
        entry = _clean_reference_entry(entries[index])
        if index + 1 < len(entries):
            next_entry = _clean_reference_entry(entries[index + 1])
            trailing_match = TRAILING_AUTHOR_GIVEN_NAME_RE.match(entry)
            if (
                trailing_match
                and LEADING_AUTHOR_SURNAME_RE.match(next_entry)
                and not YEAR_RE.search(next_entry[:80])
            ):
                entry = trailing_match.group("body").strip()
                next_entry = f"{trailing_match.group('name')} {next_entry}"
                repaired.append(_clean_reference_entry(entry))
                entries[index + 1] = next_entry
                index += 1
                continue
        repaired.append(entry)
        index += 1
    return repaired


def _extract_bracketed_references(ref_text: str, limit: Optional[int] = None) -> list[str]:
    matches = list(BRACKET_REFERENCE_RE.finditer(ref_text))
    if not matches:
        return []

    entries: list[str] = []
    selected_matches = matches[:limit] if limit is not None else matches
    for index, match in enumerate(selected_matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(ref_text)
        entry = _clean_reference_entry(ref_text[match.start():end])
        if len(entry) > 40 and re.search(r"(19|20)\d{2}", entry):
            entries.append(entry)

    return _repair_reference_boundaries(entries)


def _extract_numbered_references(ref_text: str, limit: Optional[int] = None) -> list[str]:
    normalized_ref_text = SUPERSCRIPT_NUMBERED_REFERENCE_RE.sub(r"\1. ", ref_text)
    matches = list(NUMBERED_REFERENCE_RE.finditer(normalized_ref_text))
    if not matches:
        return []

    entries: list[str] = []
    selected_matches = matches[:limit] if limit is not None else matches
    for index, match in enumerate(selected_matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_ref_text)
        entry = _clean_reference_entry(normalized_ref_text[match.start():end])
        if len(entry) > 40:
            entries.append(entry)

    return _repair_reference_boundaries(entries)


def _looks_like_reference_start(line: str) -> bool:
    if not line or PAGE_NUMBER_RE.match(line):
        return False
    if line.startswith(("URL ", "http://", "https://", "arXiv", "In ")):
        return False
    return bool(
        re.match(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'.’:-]+(?:\s|,|\.)", line)
    )


def _extract_unnumbered_references(ref_text: str, limit: Optional[int] = None) -> list[str]:
    entries: list[str] = []
    current: list[str] = []

    for raw_line in ref_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or PAGE_NUMBER_RE.match(line):
            continue

        if current and YEAR_RE.search(" ".join(current)) and _looks_like_reference_start(line):
            entries.append(_clean_reference_entry(" ".join(current)))
            current = [line]
        else:
            current.append(line)

        if limit is not None and len(entries) >= limit:
            break

    if current and (limit is None or len(entries) < limit):
        entries.append(_clean_reference_entry(" ".join(current)))

    selected_entries = entries[:limit] if limit is not None else entries
    return _repair_reference_boundaries([
        entry
        for entry in selected_entries
        if len(entry) > 40 and YEAR_RE.search(entry)
    ])


def extract_references_local(ref_text: str, limit: Optional[int] = None) -> list[str]:
    bracketed_references = _extract_bracketed_references(ref_text, limit)
    if len(bracketed_references) >= 5:
        return bracketed_references

    numbered_references = _extract_numbered_references(ref_text, limit)
    if len(numbered_references) >= 5:
        return numbered_references

    unnumbered_references = _extract_unnumbered_references(ref_text, limit)
    if unnumbered_references:
        return unnumbered_references

    if bracketed_references:
        return bracketed_references

    if numbered_references:
        return numbered_references

    entries: list[str] = []
    current: list[str] = []

    for raw_line in ref_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        if REFERENCE_START_RE.match(line) and current:
            entries.append(_clean_reference_entry(" ".join(current)))
            current = [line]
        else:
            current.append(line)

        if limit is not None and len(entries) >= limit:
            break

    if current and (limit is None or len(entries) < limit):
        entries.append(_clean_reference_entry(" ".join(current)))

    selected_entries = entries[:limit] if limit is not None else entries
    return _repair_reference_boundaries([
        entry
        for entry in selected_entries
        if len(entry) > 40 and re.search(r"(19|20)\d{2}", entry)
    ])

def extract_references(text: str):
    ref_text = extract_reference_section(text)
    if not ref_text:
        return []

    local_references = extract_references_local(ref_text)
    if local_references:
        return local_references

    return extract_references_llm(ref_text[:6000])
