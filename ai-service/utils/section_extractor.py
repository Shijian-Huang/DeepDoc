import re
from typing import Optional


SECTION_ALIASES = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "related_work": ["related work", "literature review", "prior work"],
    "method": [
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "system",
        "architecture",
    ],
    "experiment": ["experiment", "experiments", "evaluation", "setup"],
    "results": ["result", "results", "analysis", "findings"],
    "conclusion": ["conclusion", "conclusions", "discussion", "future work"],
    "references": [
        "references",
        "references and notes",
        "bibliography",
        "literature cited",
        "cited references",
    ],
}

SECTION_BUDGETS = {
    "abstract": 2200,
    "introduction": 2600,
    "related_work": 1200,
    "method": 3400,
    "experiment": 2200,
    "results": 2600,
    "conclusion": 1600,
}

SUMMARY_MODE_MAX_CHARS = {
    "paragraph": 8000,
    "standard": 12000,
    "one_page": 18000,
}

SECTION_PRIORITY = [
    "abstract",
    "introduction",
    "method",
    "experiment",
    "results",
    "conclusion",
    "related_work",
]

EMBEDDED_HEADING_PATTERNS = [
    ("references", re.compile(r"\b(?:References(?:\s+and\s+Notes)?|Bibliography|Literature\s+Cited|Cited\s+References)\s+")),
    ("conclusion", re.compile(r"\bDiscussion\s+")),
    ("conclusion", re.compile(r"\bDiscussion\s+and\s+Conclusions?\s+")),
    ("conclusion", re.compile(r"\bConclusion\.\s+")),
    ("conclusion", re.compile(r"\bConclusions?\s+and\s+Outlook\s+")),
    ("conclusion", re.compile(r"\bLimitations\.\s+")),
    ("results", re.compile(r"\bRelation to concurrent probabilistic sequence layers\.\s+")),
    ("experiment", re.compile(r"\bExperiments\s*$")),
    ("experiment", re.compile(r"\bNumerical\s+Results\s+")),
    ("results", re.compile(r"\bTheoretical\s+Results\s+")),
    ("method", re.compile(r"\bMaterials\s+and\s+Methods\s+")),
    ("method", re.compile(r"\bMethods\s+")),
    ("experiment", re.compile(r"\b4(?:\.\d+)+\s+[A-Z][A-Za-z0-9 ,:;()/'’–-]{4,}\s+")),
    ("method", re.compile(r"\b3(?:\.\d+)+\s+[A-Z][A-Za-z0-9 ,:;()/'’–⇒-]{4,}\s+")),
    ("method", re.compile(r"\b2(?:\.\d+)+\s+[A-Z][A-Za-z0-9 ,:;()/'’–-]{4,}\s+")),
    ("method", re.compile(r"\bThe design-model framework\s+")),
    ("method", re.compile(r"\bSpecializations of the design model\s+")),
]

HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\.?\s+|"
    r"[IVXLCM]+\.?\s+|"
    r"[A-Z]\.\s+"
    r")?"
)

HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVXLCM]+\.?\s+|[A-Z]\.\s+)?"
    r"[A-Z][A-Za-z0-9 &,/():'-]{2,90}\s*$"
)

INLINE_REFERENCE_START_RE = re.compile(r"^\s*\[\d+\]\s+")


def normalize_heading(line: str) -> Optional[str]:
    cleaned = HEADING_PREFIX_RE.sub("", line).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)

    for canonical, aliases in SECTION_ALIASES.items():
        if any(cleaned == alias or cleaned.startswith(alias + " ") for alias in aliases):
            return canonical

    return None


def split_embedded_heading(line: str) -> Optional[tuple[str, str, str]]:
    earliest: Optional[tuple[int, int, str, str]] = None
    for section_name, pattern in EMBEDDED_HEADING_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        if earliest is None or match.start() < earliest[0]:
            earliest = (match.start(), match.end(), section_name, match.group())

    if earliest is None:
        return None

    start, end, section_name, heading = earliest
    before = line[:start].strip()
    after = line[end:].strip()
    if section_name == "experiment" and heading.strip() == "Experiments":
        after = ""
    return before, section_name, after


def split_into_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_name = "preamble"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if INLINE_REFERENCE_START_RE.match(stripped):
            if current_lines:
                sections.setdefault(current_name, []).extend(current_lines)
            break

        normalized = normalize_heading(stripped) if HEADING_RE.match(stripped) else None

        if normalized:
            if current_lines:
                sections.setdefault(current_name, []).extend(current_lines)
            current_name = normalized
            current_lines = []
            if normalized == "references":
                break
            continue

        embedded = split_embedded_heading(line)
        if embedded:
            before, embedded_name, after = embedded
            if before:
                current_lines.append(before)
            if current_lines:
                sections.setdefault(current_name, []).extend(current_lines)
            current_name = embedded_name
            current_lines = []
            if embedded_name == "references":
                break
            if after:
                current_lines.append(after)
            continue

        current_lines.append(line)

    if current_lines:
        sections.setdefault(current_name, []).extend(current_lines)

    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if "\n".join(lines).strip()
    }


def split_pages_into_sections(pages: list[dict]) -> dict[str, dict]:
    sections: dict[str, dict] = {}
    current_name = "preamble"
    current_lines: list[str] = []
    current_pages: set[int] = set()
    reached_references = False

    def flush_current() -> None:
        if not current_lines:
            return

        text = "\n".join(current_lines).strip()
        if not text:
            return

        section = sections.setdefault(current_name, {"lines": [], "pages": set()})
        section["lines"].extend(current_lines)
        section["pages"].update(current_pages)

    for page in pages:
        page_number = int(page["page"])
        for line in page["text"].splitlines():
            stripped = line.strip()
            if INLINE_REFERENCE_START_RE.match(stripped):
                flush_current()
                reached_references = True
                break

            normalized = normalize_heading(stripped) if HEADING_RE.match(stripped) else None

            if normalized:
                flush_current()
                current_name = normalized
                current_lines = []
                current_pages = set()
                if normalized == "references":
                    reached_references = True
                    break
                continue

            embedded = split_embedded_heading(line)
            if embedded:
                before, embedded_name, after = embedded
                if before:
                    current_lines.append(before)
                    current_pages.add(page_number)
                flush_current()
                current_name = embedded_name
                current_lines = []
                current_pages = set()
                if embedded_name == "references":
                    reached_references = True
                    break
                if after:
                    current_lines.append(after)
                    current_pages.add(page_number)
                continue

            current_lines.append(line)
            current_pages.add(page_number)

        if reached_references:
            break

    flush_current()

    result = {
        name: {
            "text": "\n".join(section["lines"]).strip(),
            "pages": sorted(section["pages"]),
        }
        for name, section in sections.items()
        if "\n".join(section["lines"]).strip()
    }
    return _augment_sparse_sections(result)


def _augment_sparse_sections(sections: dict[str, dict]) -> dict[str, dict]:
    if set(sections) != {"preamble"}:
        return sections

    preamble = sections["preamble"]
    pages = preamble.get("pages") or []
    text = preamble.get("text", "")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(pages) < 3 or len(lines) < 30:
        return sections

    first_cut = max(12, len(lines) // 3)
    second_cut = max(first_cut + 12, (len(lines) * 2) // 3)
    page_first_cut = max(1, len(pages) // 3)
    page_second_cut = max(page_first_cut + 1, (len(pages) * 2) // 3)

    return {
        "preamble": {
            "text": "\n".join(lines[:6]).strip(),
            "pages": pages[:1],
        },
        "introduction": {
            "text": "\n".join(lines[6:first_cut]).strip(),
            "pages": pages[:page_first_cut],
        },
        "method": {
            "text": "\n".join(lines[first_cut:second_cut]).strip(),
            "pages": pages[page_first_cut:page_second_cut],
        },
        "conclusion": {
            "text": "\n".join(lines[second_cut:]).strip(),
            "pages": pages[page_second_cut:] or pages[-1:],
        },
    }


def _section_with_preamble_fallback(sections: dict[str, dict], section_name: str) -> Optional[dict]:
    section = sections.get(section_name)
    if section:
        return section
    if section_name == "introduction" and not sections.get("abstract"):
        return sections.get("preamble")
    return None


def _clean_section_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _clean_preamble_as_introduction(text: str) -> str:
    cleaned = _clean_section_text(text)
    metadata_patterns = [
        r"\b\*?\s*Corresponding author\.\s*Email:\s*\S+\s+",
        r"\bEmail:\s*\S+\s+",
    ]
    for pattern in metadata_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match and len(cleaned) - match.end() > 300:
            return cleaned[match.end():].strip()

    sentence_starts = [
        "Artificial intelligence",
        "Here,",
        "In this",
        "We ",
        "This ",
    ]
    candidates = [
        cleaned.find(start)
        for start in sentence_starts
        if cleaned.find(start) > 80
    ]
    if candidates:
        return cleaned[min(candidates):].strip()
    return cleaned


def _fallback_excerpt(text: str, max_chars: int) -> str:
    cleaned = _clean_section_text(text)
    if len(cleaned) <= max_chars:
        return cleaned

    head_budget = max_chars // 2
    tail_budget = max_chars - head_budget
    return f"{cleaned[:head_budget].strip()}\n\n[...]\n\n{cleaned[-tail_budget:].strip()}"


def build_summary_input(text: str, max_chars: int = 12000) -> tuple[str, list[str]]:
    sections = split_into_sections(text)
    parts: list[str] = []
    selected_sections: list[str] = []

    for section_name in SECTION_PRIORITY:
        content = sections.get(section_name)
        if not content:
            continue

        budget = SECTION_BUDGETS.get(section_name, 1500)
        excerpt = _clean_section_text(content)[:budget].strip()
        if not excerpt:
            continue

        selected_sections.append(section_name)
        parts.append(f"## {section_name.upper()}\n{excerpt}")

    if not parts:
        selected_sections = ["fallback_excerpt"]
        parts.append(f"## PAPER EXCERPTS\n{_fallback_excerpt(text, max_chars)}")

    packet = "\n\n".join(parts)
    return packet[:max_chars], selected_sections


def build_summary_input_from_pages(
    pages: list[dict],
    summary_mode: str = "standard",
) -> tuple[str, list[str], list[dict]]:
    max_chars = SUMMARY_MODE_MAX_CHARS.get(
        summary_mode,
        SUMMARY_MODE_MAX_CHARS["standard"],
    )
    sections = split_pages_into_sections(pages)
    parts: list[str] = []
    selected_sections: list[str] = []
    evidence_sources: list[dict] = []

    for section_name in SECTION_PRIORITY:
        section = _section_with_preamble_fallback(sections, section_name)
        if not section:
            continue

        budget = SECTION_BUDGETS.get(section_name, 1500)
        section_text = section["text"]
        if section_name == "introduction" and not sections.get("introduction") and not sections.get("abstract"):
            section_text = _clean_preamble_as_introduction(section_text)
        excerpt = _clean_section_text(section_text)[:budget].strip()
        if not excerpt:
            continue

        pages_text = ", ".join(str(page) for page in section["pages"])
        selected_sections.append(section_name)
        parts.append(f"## {section_name.upper()} (pages: {pages_text})\n{excerpt}")
        evidence_sources.append({
            "section": section_name,
            "pages": section["pages"],
            "excerpt": excerpt[:700],
        })

    if not parts:
        full_text = "\n".join(page["text"] for page in pages)
        selected_sections = ["fallback_excerpt"]
        fallback = _fallback_excerpt(full_text, max_chars)
        parts.append(f"## PAPER EXCERPTS\n{fallback}")
        evidence_sources.append({
            "section": "fallback_excerpt",
            "pages": [page["page"] for page in pages],
            "excerpt": fallback[:700],
        })

    packet = "\n\n".join(parts)
    return packet[:max_chars], selected_sections, evidence_sources
