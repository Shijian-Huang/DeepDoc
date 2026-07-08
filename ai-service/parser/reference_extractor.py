import re
from dataclasses import dataclass, field
from typing import Optional

from llm.summarizer import extract_references_llm


REFERENCE_SECTION_RE = re.compile(
    r"(?ims)^\s*(references(?:\s+and\s+notes)?|bibliography|literature\s+cited|cited\s+references)\s*$\n(?P<body>.*)"
)

BRACKET_REFERENCE_RE = re.compile(r"(?m)^\s*\[(\d+)\][ \t]+")
NUMBERED_REFERENCE_RE = re.compile(r"(?m)^\s*(\d{1,3})[.)][ \t]*(?=[A-Z])")
BRACKET_REFERENCE_ANYWHERE_RE = re.compile(r"(?<!\w)\[(\d{1,3})\][ \t]+")

REFERENCE_START_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d{1,3}\.|\d{1,3}\)|[A-Z][A-Za-z'-]+,\s+[A-Z])"
)

TRAILING_SECTION_RE = re.compile(
    r"(?im)^\s*(appendix|appendices|acknowledg(e)?ments?|code\s+availability|data\s+availability|supplementary\s+materials?)\s*$"
)

APPENDIX_HEADING_RE = re.compile(r"^(appendix|appendices)\b", re.IGNORECASE)
APPENDIX_LETTER_HEADING_RE = re.compile(r"^[A-Z]\.\s+[^,]{4,80}$")
APPENDIX_TITLE_RE = re.compile(r"^[A-Z].{4,}$")
YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
APPENDIX_MARKER_RE = re.compile(r"^[A-Z]$")
REFERENCE_CONTINUATION_RE = re.compile(
    r"(?i)^(arxiv|preprint|journal|proceedings|in\s+|pp\.|pages?\b|url\b|doi\b|"
    r"conference|workshop|transactions|annual review|phd thesis|university\b)"
)
PAPER_HEADER_HINT_RE = re.compile(
    r"(?i)\b(arxiv|proceedings|transactions|conference|workshop|journal|"
    r"investigating|towards?|toward|learning|model|models|large language model)\b"
)
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


@dataclass
class ReferenceExtractionResult:
    references: list[str]
    method: str = "none"
    repaired: bool = False
    low_confidence: bool = False
    expected_count: int = 0
    section_char_count: int = 0
    notes: list[str] = field(default_factory=list)


def extract_reference_section(text: str) -> str:
    return _extract_reference_section(text, relaxed=False)


def extract_reference_section_relaxed(text: str) -> str:
    return _extract_reference_section(text, relaxed=True)


def _extract_reference_section(text: str, relaxed: bool = False) -> str:
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
    normalized_line_counts: dict[str, int] = {}
    for raw_line in raw_lines:
        normalized_line = re.sub(r"\s+", " ", raw_line).strip()
        if normalized_line:
            normalized_line_counts[normalized_line] = normalized_line_counts.get(normalized_line, 0) + 1

    lines: list[str] = []
    for index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        normalized_line = re.sub(r"\s+", " ", line)
        if (
            len(normalized_line) > 30
            and normalized_line_counts.get(normalized_line, 0) >= 2
            and PAPER_HEADER_HINT_RE.search(normalized_line)
            and not YEAR_RE.search(normalized_line)
            and not re.search(r"\b(URL|doi|arXiv)\b", normalized_line, re.IGNORECASE)
        ):
            continue
        if lines and APPENDIX_HEADING_RE.match(line):
            break
        next_line = ""
        for candidate in raw_lines[index + 1:]:
            if candidate.strip():
                next_line = candidate.strip()
                break
        if (
            not relaxed
            and lines
            and APPENDIX_LETTER_HEADING_RE.match(line)
            and not REFERENCE_CONTINUATION_RE.match(next_line)
            and not YEAR_RE.search(next_line[:80])
        ):
            break
        if not relaxed and lines and APPENDIX_MARKER_RE.match(line) and APPENDIX_TITLE_RE.match(next_line):
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


def _extract_inline_bracketed_references(ref_text: str, limit: Optional[int] = None) -> list[str]:
    matches = list(BRACKET_REFERENCE_ANYWHERE_RE.finditer(ref_text))
    if not matches:
        return []

    entries: list[str] = []
    selected_matches = matches[:limit] if limit is not None else matches
    for index, match in enumerate(selected_matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(ref_text)
        entry = _clean_reference_entry(ref_text[match.start():end])
        if len(entry) > 40 and YEAR_RE.search(entry):
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

    inline_bracketed_references = _extract_inline_bracketed_references(ref_text, limit)
    if len(inline_bracketed_references) >= 5:
        return inline_bracketed_references

    numbered_references = _extract_numbered_references(ref_text, limit)
    if len(numbered_references) >= 5:
        return numbered_references

    unnumbered_references = _extract_unnumbered_references(ref_text, limit)
    if unnumbered_references:
        return unnumbered_references

    if bracketed_references:
        return bracketed_references

    if inline_bracketed_references:
        return inline_bracketed_references

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


def _expected_marker_count(ref_text: str) -> int:
    bracket_numbers = [
        int(match.group(1))
        for match in BRACKET_REFERENCE_ANYWHERE_RE.finditer(ref_text)
        if int(match.group(1)) <= 200
    ]
    if bracket_numbers:
        return max(bracket_numbers)

    numbered_text = SUPERSCRIPT_NUMBERED_REFERENCE_RE.sub(r"\1. ", ref_text)
    numbered_matches = [
        int(match.group(1))
        for match in NUMBERED_REFERENCE_RE.finditer(numbered_text)
        if int(match.group(1)) <= 200
    ]
    return max(numbered_matches) if numbered_matches else 0


def _reference_quality_notes(ref_text: str, references: list[str]) -> list[str]:
    notes: list[str] = []
    expected_count = _expected_marker_count(ref_text)
    if not references:
        notes.append("no_references")
        return notes
    if len(ref_text) > 1000 and len(references) < 5:
        notes.append("few_references_for_long_section")
    if expected_count >= 5 and len(references) < max(3, expected_count // 2):
        notes.append("parsed_count_far_below_numbered_markers")
    if any(len(reference) > 2200 for reference in references):
        notes.append("oversized_reference_entry")
    if any(BRACKET_REFERENCE_ANYWHERE_RE.search(reference[12:]) for reference in references):
        notes.append("merged_bracketed_references")
    return notes


def extract_references_with_diagnostics(text: str, allow_llm: bool = True) -> ReferenceExtractionResult:
    ref_text = extract_reference_section(text)
    if not ref_text:
        return ReferenceExtractionResult(references=[], notes=["missing_reference_section"])

    local_references = extract_references_local(ref_text)
    expected_count = _expected_marker_count(ref_text)
    notes = _reference_quality_notes(ref_text, local_references)
    if local_references and not notes:
        return ReferenceExtractionResult(
            references=local_references,
            method="local",
            expected_count=expected_count,
            section_char_count=len(ref_text),
        )

    relaxed_ref_text = extract_reference_section_relaxed(text)
    if relaxed_ref_text and relaxed_ref_text != ref_text:
        relaxed_references = extract_references_local(relaxed_ref_text)
        relaxed_notes = _reference_quality_notes(relaxed_ref_text, relaxed_references)
        if relaxed_references and (
            not relaxed_notes
            or len(relaxed_references) > len(local_references)
        ):
            return ReferenceExtractionResult(
                references=relaxed_references,
                method="local_relaxed",
                repaired=True,
                low_confidence=bool(relaxed_notes),
                expected_count=_expected_marker_count(relaxed_ref_text),
                section_char_count=len(relaxed_ref_text),
                notes=relaxed_notes,
            )

    if allow_llm:
        llm_references = extract_references_llm((relaxed_ref_text or ref_text)[:12000])
        llm_notes = _reference_quality_notes(relaxed_ref_text or ref_text, llm_references)
        if llm_references and len(llm_references) > len(local_references):
            return ReferenceExtractionResult(
                references=llm_references,
                method="llm_repair",
                repaired=True,
                low_confidence=bool(llm_notes),
                expected_count=_expected_marker_count(relaxed_ref_text or ref_text),
                section_char_count=len(relaxed_ref_text or ref_text),
                notes=llm_notes,
            )

    return ReferenceExtractionResult(
        references=local_references,
        method="local_low_confidence" if local_references else "none",
        low_confidence=True,
        expected_count=expected_count,
        section_char_count=len(ref_text),
        notes=notes or ["unrepaired_low_confidence"],
    )


def extract_references(text: str):
    return extract_references_with_diagnostics(text, allow_llm=False).references
