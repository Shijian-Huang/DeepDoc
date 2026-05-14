import re

from llm.summarizer import extract_references_llm


REFERENCE_SECTION_RE = re.compile(
    r"(?ims)^\s*(references|bibliography)\s*$\n(?P<body>.*)"
)

BRACKET_REFERENCE_RE = re.compile(r"(?m)^\s*\[(\d+)\]\s+")

REFERENCE_START_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d{1,3}\.|\d{1,3}\)|[A-Z][A-Za-z'-]+,\s+[A-Z])"
)

TRAILING_SECTION_RE = re.compile(
    r"(?im)^\s*(appendix|appendices|acknowledg(e)?ments?)\s*$"
)


def extract_reference_section(text: str) -> str:
    match = REFERENCE_SECTION_RE.search(text)
    if not match:
        return ""

    ref_text = match.group("body")
    trailing_match = TRAILING_SECTION_RE.search(ref_text)
    if trailing_match:
        ref_text = ref_text[:trailing_match.start()]

    return ref_text.strip()


def _clean_reference_entry(entry: str) -> str:
    entry = re.sub(r"\s+", " ", entry).strip()
    entry = re.sub(r"(?<=\w)-\s+(?=\w)", "", entry)
    entry = re.sub(r"https:\s+//", "https://", entry)
    return entry


def _extract_bracketed_references(ref_text: str, limit: int) -> list[str]:
    matches = list(BRACKET_REFERENCE_RE.finditer(ref_text))
    if not matches:
        return []

    entries: list[str] = []
    for index, match in enumerate(matches[:limit]):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(ref_text)
        entry = _clean_reference_entry(ref_text[match.start():end])
        if len(entry) > 40 and re.search(r"(19|20)\d{2}", entry):
            entries.append(entry)

    return entries


def extract_references_local(ref_text: str, limit: int = 10) -> list[str]:
    bracketed_references = _extract_bracketed_references(ref_text, limit)
    if bracketed_references:
        return bracketed_references

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

        if len(entries) >= limit:
            break

    if current and len(entries) < limit:
        entries.append(_clean_reference_entry(" ".join(current)))

    return [
        entry
        for entry in entries[:limit]
        if len(entry) > 40 and re.search(r"(19|20)\d{2}", entry)
    ]

def extract_references(text: str):
    ref_text = extract_reference_section(text)
    if not ref_text:
        return []

    local_references = extract_references_local(ref_text)
    if local_references:
        return local_references

    return extract_references_llm(ref_text[:6000])
