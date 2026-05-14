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
    "results": ["result", "results", "discussion", "analysis", "findings"],
    "conclusion": ["conclusion", "conclusions", "future work"],
    "references": ["references", "bibliography"],
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

SECTION_PRIORITY = [
    "abstract",
    "introduction",
    "method",
    "experiment",
    "results",
    "conclusion",
    "related_work",
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


def normalize_heading(line: str) -> Optional[str]:
    cleaned = HEADING_PREFIX_RE.sub("", line).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)

    for canonical, aliases in SECTION_ALIASES.items():
        if any(cleaned == alias or cleaned.startswith(alias + " ") for alias in aliases):
            return canonical

    return None


def split_into_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_name = "preamble"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        normalized = normalize_heading(stripped) if HEADING_RE.match(stripped) else None

        if normalized:
            if current_lines:
                sections.setdefault(current_name, []).extend(current_lines)
            current_name = normalized
            current_lines = []
            if normalized == "references":
                break
            continue

        current_lines.append(line)

    if current_lines:
        sections.setdefault(current_name, []).extend(current_lines)

    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if "\n".join(lines).strip()
    }


def _clean_section_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


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
