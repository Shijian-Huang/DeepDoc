import os
import json
import re
import time
from pathlib import Path
from typing import Any, Optional
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv 

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=30000)
)

gemini_models = ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite"]
request_interval_seconds = 4.1
last_request_at = 0.0


def is_gemini_configured() -> bool:
    return bool(api_key)


def gemini_configuration_error() -> str:
    return "GEMINI_API_KEY is not configured. Add it to ai-service/.env or the server environment."


def require_gemini_api_key() -> None:
    if not is_gemini_configured():
        raise RuntimeError(gemini_configuration_error())

SUMMARY_MODE_INSTRUCTIONS = {
    "paragraph": (
        "Task: Should I read this paper? "
        "Optimize for fast understanding. "
        "The reader should decide whether the paper is worth reading within 30 seconds. "
        "Write a concise Research Snapshot in 2 short paragraphs, 100-150 words total. "
        "Paragraph 1 should explain the paper's problem and why it matters. "
        "Paragraph 2 should explain the core idea or approach and the main takeaway. "
        "Avoid implementation details, background discussion, and excessive methodology."
    ),
    "standard": (
        "Task: Help me understand this paper. "
        "Optimize for comprehension. "
        "The reader should understand the paper without reading the original PDF. "
        "Write the overview in 4 well-balanced paragraphs, 220-300 words total. "
        "Structure the summary as: 1. context and motivation; 2. the problem being addressed; "
        "3. the proposed approach and key findings; 4. practical implications and final takeaway. "
        "The summary should read like an executive brief for a technical reader."
    ),
    "one_page": (
        "Task: Help me study this paper. "
        "Optimize for study. "
        "The reader should feel prepared to discuss the paper in a research meeting after reading this summary. "
        "Write a detailed research brief in 5-6 structured paragraphs, 450-700 words total. "
        "Cover context, research problem, technical approach, major findings, limitations if discussed, "
        "and broader implications and takeaway. Include enough technical detail for readers who want "
        "to understand the paper before reading the full text, but avoid reproducing the paper section by section."
    ),
}

SUMMARY_MODE_MIN_WORDS = {
    "paragraph": 100,
    "standard": 220,
    "one_page": 450,
}

SUMMARY_MODE_TARGET_WORDS = {
    "paragraph": "100-150",
    "standard": "220-300",
    "one_page": "450-700",
}

SUMMARY_MODE_TARGET_PARAGRAPHS = {
    "paragraph": 2,
    "standard": 4,
    "one_page": 5,
}

def wait_for_rate_limit():
    global last_request_at

    elapsed = time.monotonic() - last_request_at
    if elapsed < request_interval_seconds:
        time.sleep(request_interval_seconds - elapsed)

    last_request_at = time.monotonic()

def extract_json(raw_text: str) -> str:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if match:
        return match.group()

    return cleaned

def generate_json(prompt: str):
    require_gemini_api_key()
    last_raw_text = ""

    for model in gemini_models:
        wait_for_rate_limit()

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
        except (errors.ClientError, errors.ServerError) as error:
            last_raw_text = str(error)
            continue

        raw_text = response.text or ""
        cleaned = extract_json(raw_text)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            last_raw_text = raw_text

    raise json.JSONDecodeError("Could not parse model response as JSON", last_raw_text, 0)

def summarize_chunk(chunk: str):
    prompt = f"""
    You are analyzing a section of a research paper.
    Only use information from the provided text. Do not invent details.

    Focus on:
    - Problem
    - Method
    - Key findings

    Return ONLY valid JSON:
    {{
      "summary": "...",
      "key_points": ["...", "..."]
    }}

    Text:
    {chunk}
    """

    try:
        return generate_json(prompt)
    except json.JSONDecodeError as error:
        return {
            "summary":"",
            "key_points":[],
            "error": error.doc
        }

def summarize_document(chunk_summaries: list):
    combined = "\n\n".join([
        f"Section {i + 1}:\n{c.get('summary', '')}"
        for i, c in enumerate(chunk_summaries)
        ])

    prompt = f"""
    You are writing an overall analysis of a research paper based on section summaries.
    Each section corresponds to a different part of the paper.
    Only use information from the provided section summaries. Do not invent details.

    Focus on:
    - Main problem
    - Core method
    - Key findings
    - Main contributions

    Return ONLY valid JSON:
    {{
      "summary": "...",
      "key_ideas": ["...", "..."],
      "contributions": ["...", "..."]
    }}

    Section summaries:
    {combined}
    """

    try:
        return generate_json(prompt)
    except json.JSONDecodeError as error:
        return {
            "summary":"Document summary failed.",
            "key_ideas":[],
            "contributions": [],
            "error": error.doc
        }

def normalize_summary_mode(summary_mode: str) -> str:
    if summary_mode in SUMMARY_MODE_INSTRUCTIONS:
        return summary_mode

    return "standard"


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _normalize_summary_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    raw_paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs: list[str] = []
    for paragraph in raw_paragraphs:
        cleaned = re.sub(r"[ \t]+", " ", paragraph)
        cleaned = re.sub(r"\n+", " ", cleaned).strip()
        if cleaned:
            paragraphs.append(cleaned)

    if len(paragraphs) <= 1:
        return re.sub(r"\s+", " ", text).strip()

    return "\n\n".join(paragraphs)


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _chunk_sentences(sentences: list[str], paragraph_count: int) -> list[str]:
    if paragraph_count <= 1 or len(sentences) < paragraph_count:
        return [" ".join(sentences).strip()] if sentences else []

    chunks: list[str] = []
    total = len(sentences)
    for index in range(paragraph_count):
        start = round(index * total / paragraph_count)
        end = round((index + 1) * total / paragraph_count)
        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _chunk_words(text: str, paragraph_count: int) -> list[str]:
    words = re.findall(r"\S+", str(text or ""))
    if paragraph_count <= 1 or len(words) < paragraph_count * 12:
        return [re.sub(r"\s+", " ", text).strip()] if text else []

    chunks: list[str] = []
    total = len(words)
    for index in range(paragraph_count):
        start = round(index * total / paragraph_count)
        end = round((index + 1) * total / paragraph_count)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _enforce_summary_paragraphs(summary: str, summary_mode: str) -> str:
    target_paragraphs = SUMMARY_MODE_TARGET_PARAGRAPHS.get(summary_mode, 4)
    existing_paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", summary)
        if paragraph.strip()
    ]
    if len(existing_paragraphs) >= target_paragraphs:
        return "\n\n".join(existing_paragraphs)

    sentences = _split_sentences(" ".join(existing_paragraphs) if existing_paragraphs else summary)
    flat_summary = " ".join(existing_paragraphs) if existing_paragraphs else summary

    paragraph_count = target_paragraphs
    if summary_mode == "one_page" and len(sentences) >= 12:
        paragraph_count = 6

    paragraphs = (
        _chunk_sentences(sentences, paragraph_count)
        if len(sentences) >= target_paragraphs
        else _chunk_words(flat_summary, paragraph_count)
    )
    if len(paragraphs) < target_paragraphs:
        return summary
    return "\n\n".join(paragraphs)


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value:
        items = [value]
    else:
        items = []

    texts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("claim")
                or item.get("title")
                or item.get("summary")
                or ""
            )
        else:
            text = str(item or "")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            texts.append(text)
    return texts


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "their",
        "paper", "study", "approach", "method", "results", "show", "shows",
        "using", "used", "can", "are", "was", "were", "has", "have",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
        if token not in stopwords
    }


def _is_near_duplicate(left: str, right: str) -> bool:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return left.strip().lower() == right.strip().lower()
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.78


def _dedupe_text_list(items: list[str], blocked_items: list[str] | None = None, limit: int = 6) -> list[str]:
    blocked_items = blocked_items or []
    deduped: list[str] = []
    for item in items:
        if any(_is_near_duplicate(item, blocked) for blocked in blocked_items):
            continue
        if any(_is_near_duplicate(item, existing) for existing in deduped):
            continue
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _normalize_pages(value: Any) -> list[int]:
    raw_pages = value if isinstance(value, list) else [value]
    pages: list[int] = []
    for raw_page in raw_pages:
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            continue
        if page > 0 and page not in pages:
            pages.append(page)
    return pages


def normalize_research_summary_result(result: dict, summary_mode: str) -> dict:
    normalized = dict(result or {})
    summary = _normalize_summary_text(normalized.get("summary"))
    summary = _enforce_summary_paragraphs(summary, summary_mode)
    normalized["summary"] = summary
    normalized["summary_word_count"] = count_words(summary)

    key_ideas = _dedupe_text_list(_as_text_list(normalized.get("key_ideas")), limit=6)
    contributions = _dedupe_text_list(
        _as_text_list(normalized.get("contributions")),
        blocked_items=key_ideas,
        limit=5,
    )
    normalized["key_ideas"] = key_ideas
    normalized["contributions"] = contributions
    evidence_items: list[dict] = []
    evidence = normalized.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            claim = re.sub(r"\s+", " ", str(item.get("claim") or item.get("summary") or "")).strip()
            if not claim:
                continue
            section = re.sub(r"\s+", "_", str(item.get("section") or "unknown").strip().lower()) or "unknown"
            pages = _normalize_pages(item.get("pages") or item.get("page_numbers") or item.get("page"))
            evidence_items.append({
                "claim": claim,
                "section": section,
                "pages": pages,
            })
            if len(evidence_items) >= 6:
                break
    normalized["evidence"] = evidence_items
    return normalized


def needs_summary_expansion(result: dict, summary_mode: str) -> bool:
    summary = result.get("summary", "")
    return count_words(summary) < SUMMARY_MODE_MIN_WORDS[summary_mode]


def _evidence_sections_from_packet(evidence_packet: str) -> dict[str, list[int]]:
    sections: dict[str, list[int]] = {}
    pattern = re.compile(
        r"^##\s+([A-Z_]+)(?:\s+\(pages:\s*([^)]+)\))?",
        re.MULTILINE,
    )
    for match in pattern.finditer(evidence_packet):
        section = match.group(1).strip().lower()
        pages_text = match.group(2) or ""
        pages = [
            int(value)
            for value in re.findall(r"\d+", pages_text)
        ]
        sections[section] = pages
    return sections


def _normalize_evidence_sources(result: dict, evidence_packet: str) -> None:
    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        return

    sections = _evidence_sections_from_packet(evidence_packet)
    empirical_terms = re.compile(
        r"\b("
        r"performance|benchmark|accuracy|retrieval|recall|ruler|mqar|"
        r"distill|distilling|distillation|gain|gains|improve|improves|"
        r"improved|outperform|baseline|ablation|experiment|robustness"
        r")\b",
        re.IGNORECASE,
    )

    target_section = None
    if sections.get("experiment"):
        target_section = "experiment"
    elif sections.get("results"):
        target_section = "results"

    if not target_section:
        return

    for item in evidence:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "")
        section = str(item.get("section") or "").lower()
        if section == "abstract" and empirical_terms.search(claim):
            item["section"] = target_section
            item["pages"] = sections.get(target_section, item.get("pages") or [])


def build_research_summary_prompt(
    evidence_packet: str,
    summary_mode: str,
    retry_word_count: Optional[int] = None,
) -> str:
    length_instruction = SUMMARY_MODE_INSTRUCTIONS[summary_mode]
    retry_instruction = ""
    if retry_word_count is not None:
        retry_instruction = f"""
    Previous attempt was too short at {retry_word_count} words.
    Rewrite and expand the summary field to {SUMMARY_MODE_TARGET_WORDS[summary_mode]} words.
    Keep the summary faithful to the provided paper text.
    """

    return f"""
    You are analyzing a research paper from selected high-value sections.
    Only use information from the provided text. Do not invent details.

    Summary mode: {summary_mode}
    Length requirement: {length_instruction}
    Treat the selected mode as a distinct reader task, not as a short/medium/long version of the same summary.
    The mode's task and optimization goal are more important than merely hitting a word count.
    The length requirement applies only to the "summary" field.
    Do not count key_ideas, contributions, references, evidence, or summary_word_count toward the word count.
    Do not generate one continuous block of text. Use well-balanced paragraphs with clear logical progression.
    Each paragraph should focus on one primary purpose.
    {retry_instruction}

    Information architecture:
    - The "summary" field is an Overview. It should tell the overall narrative of the paper:
      context, why the problem matters, the broad approach, and the final takeaway.
    - Do not repeat the key ideas or contributions in detail inside the Overview.
      Those belong in the dedicated "key_ideas" and "contributions" fields.
    - "key_ideas" should capture the most important conceptual ideas needed to understand the paper.
      Avoid phrasing these as novelty claims.
    - "contributions" should capture what is genuinely new or added by the paper.
      Avoid repeating general background, motivation, or the same wording used in key_ideas.
    - Keep all fields evidence-grounded.

    Evidence selection rules:
    - Prefer specific evidence from method, experiment, results, or conclusion sections.
    - Use abstract evidence only for high-level framing or definitions that are not repeated in later sections.
    - For empirical claims about performance, benchmarks, accuracy, retrieval, robustness, ablations, distillation,
      or comparisons against baselines, choose experiment or results pages rather than the abstract.
    - Each evidence claim should be anchored to the most specific section and page range available in the paper text.
    - Include 4-6 evidence items when enough grounded claims are available.
    Return ONLY valid JSON:
    {{
      "summary": "...",
      "summary_word_count": 0,
      "key_ideas": ["...", "..."],
      "contributions": ["...", "..."],
      "evidence": [
        {{
          "claim": "...",
          "section": "abstract|introduction|method|experiment|results|conclusion|related_work",
          "pages": [1, 2]
        }}
      ]
    }}

    Paper text:
    {evidence_packet}
    """


def summarize_research_paper(evidence_packet: str, summary_mode: str = "standard"):
    normalized_mode = normalize_summary_mode(summary_mode)
    prompt = build_research_summary_prompt(evidence_packet, normalized_mode)

    try:
        result = generate_json(prompt)
        result = normalize_research_summary_result(result, normalized_mode)
        word_count = result["summary_word_count"]
        _normalize_evidence_sources(result, evidence_packet)

        if needs_summary_expansion(result, normalized_mode):
            retry_prompt = build_research_summary_prompt(
                evidence_packet,
                normalized_mode,
                retry_word_count=word_count,
            )
            retry_result = generate_json(retry_prompt)
            retry_result = normalize_research_summary_result(retry_result, normalized_mode)
            _normalize_evidence_sources(retry_result, evidence_packet)
            return retry_result

        return result
    except json.JSONDecodeError as error:
        return {
            "summary": "Document summary failed.",
            "summary_word_count": 0,
            "key_ideas": [],
            "contributions": [],
            "evidence": [],
            "error": error.doc
        }


def compact_video_sources(sources: list, limit: int = 6, excerpt_chars: int = 420) -> list[dict]:
    compacted: list[dict] = []
    for source in sources[:limit]:
        if not isinstance(source, dict):
            continue
        compacted.append({
            "section": source.get("section", ""),
            "pages": source.get("pages", []),
            "excerpt": str(source.get("excerpt", ""))[:excerpt_chars],
        })
    return compacted


def rank_video_evidence(evidence: list) -> list[dict]:
    if not isinstance(evidence, list):
        return []

    general_terms = {
        "improve": 7,
        "improved": 7,
        "improvement": 7,
        "increase": 6,
        "increased": 6,
        "reduce": 6,
        "reduced": 6,
        "decrease": 6,
        "decreased": 6,
        "higher": 5,
        "lower": 5,
        "better": 5,
        "worse": 5,
        "significant": 6,
        "statistically": 6,
        "outperform": 8,
        "outperformed": 8,
        "more effective": 8,
        "more effectively": 8,
        "less effective": 7,
        "accuracy": 5,
        "precision": 5,
        "recall": 5,
        "sensitivity": 5,
        "specificity": 5,
        "performance": 5,
        "result": 4,
        "finding": 4,
        "experiment": 4,
        "study": 3,
        "participants": 3,
        "dataset": 3,
        "evaluation": 3,
        "risk": 4,
        "limitation": 4,
        "challenge": 4,
        "trade-off": 4,
        "privacy": 3,
        "safety": 3,
        "security": 3,
        "clinical": 3,
        "patient": 3,
        "education": 3,
        "learning": 3,
        "software": 3,
    }
    weighted_terms = {
        "commercial": 7,
        "vulnerabil": 7,
        "malware": 6,
        "detection": 5,
        "fine-tun": 4,
        "tunability": 4,
        "dual-use": 4,
        "privacy": 3,
        "legal": 3,
        "compliance": 3,
    }

    ranked: list[tuple[int, int, dict]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).lower()
        section = str(item.get("section", "")).lower()
        score = sum(weight for term, weight in general_terms.items() if term in claim)
        score += sum(weight for term, weight in weighted_terms.items() if term in claim)
        if re.search(r"\b\d+(\.\d+)?\s*(%|percent|x|times|fold|participants|samples|cases|papers|studies)\b", claim):
            score += 8
        elif re.search(r"\b\d+(\.\d+)?\b", claim):
            score += 4
        if re.search(r"\b(compared|versus|vs\.?|relative to|than|baseline|control group)\b", claim):
            score += 6
        if section in {"results", "experiment", "experiments", "evaluation"}:
            score += 5
        elif section in {"abstract", "conclusion"}:
            score += 3
        elif section in {"introduction", "method"}:
            score += 2
        if item.get("pages"):
            score += 2
        ranked.append((score, -index, item))

    ranked.sort(reverse=True)
    return [item for _, _, item in ranked]


def soften_unsupported_causality(text: str) -> str:
    softened = re.sub(r"\bis driven by\b", "is associated with", text, flags=re.IGNORECASE)
    softened = re.sub(r"\bare driven by\b", "are associated with", softened, flags=re.IGNORECASE)
    softened = re.sub(r"\bwas driven by\b", "was associated with", softened, flags=re.IGNORECASE)
    softened = re.sub(r"\bwere driven by\b", "were associated with", softened, flags=re.IGNORECASE)
    softened = re.sub(r"\bdrives the shift\b", "is part of the shift", softened, flags=re.IGNORECASE)
    softened = re.sub(r"\bdriving the shift\b", "part of the shift", softened, flags=re.IGNORECASE)
    return softened


def normalize_video_scene(scene: dict) -> None:
    for key in ["heading", "voiceover", "visual_note"]:
        if isinstance(scene.get(key), str):
            scene[key] = soften_unsupported_causality(scene[key])

    bullets = scene.get("bullets")
    if isinstance(bullets, list):
        scene["bullets"] = [
            soften_unsupported_causality(str(bullet))
            for bullet in bullets
        ]


def build_video_scene_roles(slide_count: int) -> list[str]:
    profiles = {
        5: [
            "hook",
            "problem",
            "core_idea",
            "implication",
            "takeaway",
        ],
        8: [
            "hook",
            "problem",
            "method_overview",
            "core_idea",
            "key_finding",
            "evidence",
            "limitation",
            "takeaway",
        ],
        10: [
            "hook",
            "problem",
            "why_it_matters",
            "core_idea",
            "method_overview",
            "key_finding",
            "evidence",
            "limitation",
            "implication",
            "takeaway",
        ],
        15: [
            "hook",
            "problem",
            "why_it_matters",
            "core_idea",
            "method_overview",
            "method",
            "mechanism",
            "example",
            "key_finding",
            "evidence",
            "comparison",
            "supporting_detail",
            "limitation",
            "implication",
            "takeaway",
        ],
    }
    if slide_count in profiles:
        return profiles[slide_count]

    long_roles = profiles[15]
    if slide_count < 5:
        return profiles[5][:slide_count - 1] + ["takeaway"]
    if slide_count < 8:
        return profiles[8][:slide_count - 1] + ["takeaway"]
    if slide_count < 10:
        return profiles[10][:slide_count - 1] + ["takeaway"]
    if slide_count < 15:
        return long_roles[:slide_count - 1] + ["takeaway"]
    return long_roles + ["supporting_detail"] * (slide_count - len(long_roles))


def video_slide_profile(slide_count: int) -> dict[str, str]:
    if slide_count <= 5:
        return {
            "name": "brief",
            "duration_range": "60-85",
            "evidence_rule": "Use 2-4 evidence-backed scenes. Reuse no evidence claim unless it is the strongest evidence.",
            "structure_rule": "Compress ruthlessly into a story: hook, problem, core insight, implication, takeaway. Skip implementation detail unless it is the story.",
        }
    if slide_count <= 8:
        return {
            "name": "balanced",
            "duration_range": "95-130",
            "evidence_rule": "Use 4-6 evidence-backed scenes. Reuse the strongest evidence at most once.",
            "structure_rule": "Build a compact talk arc: hook, problem, method, core idea, finding, evidence, limitation, takeaway. Each scene must introduce a new beat.",
        }
    if slide_count <= 10:
        return {
            "name": "deep dive",
            "duration_range": "130-175",
            "evidence_rule": "Use 5-7 evidence-backed scenes. Prefer different sections for method, results, comparison, and boundary slides.",
            "structure_rule": "Use a research-group presentation arc. Explain why the paper matters, then the core idea, method, finding, evidence, limitation, implication, and takeaway.",
        }
    return {
        "name": "lecture",
        "duration_range": "190-250",
        "evidence_rule": "Use 8-11 evidence-backed scenes. Reuse the same evidence claim at most twice, and only when connecting setup to takeaway.",
        "structure_rule": "Treat this as a mini conference talk. Add method, mechanism, example, comparison, limitation, and implication only when each beat moves the story forward.",
    }


def max_video_evidence_reuse(slide_count: int) -> int:
    return 1 if slide_count <= 10 else 2


def make_fallback_video_scene(index: int, role: str, analysis_result: dict, evidence: list[dict]) -> dict:
    document_summary = analysis_result.get("document_summary", {})
    key_ideas = document_summary.get("key_ideas", [])
    contributions = document_summary.get("contributions", [])
    summary = document_summary.get("summary", "")
    source_items = key_ideas or contributions or ([summary] if summary else [])
    seed = source_items[(index - 1) % len(source_items)] if source_items else "The paper develops a focused research argument."
    heading = str(seed).split(".")[0][:90] or f"Slide {index}"
    scene_evidence = evidence[(index - 1) % len(evidence)] if evidence else {}
    return {
        "scene_number": index,
        "role": role,
        "heading": heading,
        "bullets": [heading[:42], "Paper-backed point"],
        "voiceover": str(seed),
        "evidence": scene_evidence,
        "visual_type": "takeaway" if role == "takeaway" else "evidence_card",
        "visual_note": "Clean Apple-style text slide with one focused idea.",
    }


def _first_sentence(text: str, max_chars: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentence = ""
    for candidate in sentences:
        if len(candidate) >= 45:
            sentence = candidate
            break
    if not sentence:
        sentence = sentences[0] if sentences else cleaned
    if len(sentence) > max_chars:
        return ""
    if _looks_truncated(sentence):
        return ""
    return sentence


def source_video_evidence_candidates(sources: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        section = str(source.get("section") or "").strip().lower()
        excerpt = str(source.get("excerpt") or "")
        if section not in {"abstract", "introduction", "method", "conclusion"}:
            continue
        if re.search(r"\b(recent|concurrent) works\b", excerpt, re.IGNORECASE):
            continue
        claim = _first_sentence(excerpt)
        if len(claim) < 50:
            continue
        candidates.append({
            "claim": claim,
            "section": section,
            "pages": source.get("pages") or [],
            "source": "section_excerpt",
        })
    return candidates


def _evidence_key(evidence_item: Any) -> str:
    if not isinstance(evidence_item, dict):
        return ""
    return "|".join([
        str(evidence_item.get("section") or "").strip().lower(),
        ",".join(str(page) for page in evidence_item.get("pages") or []),
        str(evidence_item.get("claim") or "").strip().lower()[:120],
    ])


def _looks_truncated(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    if re.search(r"\b(et|or|and|the|a|an|of|in|to|for|with|whose|when|while|where|which)\.$", cleaned, re.IGNORECASE):
        return True
    if cleaned.endswith((".", "!", "?", ":", ";", ")", "]", '"', "'")):
        return False
    return len(cleaned.split()) >= 8


VIDEO_EVIDENCE_STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "a", "an", "as", "by", "for", "with",
    "from", "this", "that", "these", "those", "model", "models", "layer", "layers",
    "paper", "research", "sequence", "sequences", "architecture", "architectures",
    "bayesian",
}


ROLE_EVIDENCE_SECTIONS = {
    "hook": {"abstract", "introduction", "experiment", "results", "conclusion"},
    "surprising_finding": {"experiment", "results", "abstract"},
    "strongest_evidence": {"experiment", "results", "abstract"},
    "evidence": {"experiment", "results", "method", "abstract"},
    "key_finding": {"experiment", "results", "abstract", "conclusion"},
    "key_evidence": {"experiment", "results"},
    "results": {"experiment", "results"},
    "problem": {"introduction", "method", "experiment", "results"},
    "why_it_matters": {"abstract", "introduction", "method"},
    "core_idea": {"abstract", "introduction", "method"},
    "method": {"abstract", "introduction", "method"},
    "method_overview": {"abstract", "introduction", "method"},
    "technical_insight": {"abstract", "method"},
    "mechanism": {"abstract", "method"},
    "example": {"method", "experiment", "results", "abstract"},
    "comparison": {"abstract", "introduction", "method", "experiment", "results"},
    "limitation": {"introduction", "method", "experiment", "results", "conclusion", "abstract"},
    "implication": {"abstract", "introduction", "method", "experiment", "results", "conclusion"},
    "supporting_detail": {"abstract", "introduction", "method", "experiment", "results", "conclusion"},
    "boundary": {"introduction", "method", "experiment", "results", "conclusion", "abstract"},
    "design_principle": {"abstract", "method", "conclusion"},
    "takeaway": {"abstract", "experiment", "results", "conclusion"},
}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", str(text).lower())
        if token not in VIDEO_EVIDENCE_STOPWORDS
    }


def _scene_text(scene: dict) -> str:
    bullets = scene.get("bullets") if isinstance(scene.get("bullets"), list) else []
    return " ".join([
        str(scene.get("role") or ""),
        str(scene.get("heading") or ""),
        " ".join(str(item) for item in bullets),
        str(scene.get("voiceover") or ""),
    ])


def _video_evidence_matches_scene(scene: dict, evidence_item: dict) -> bool:
    role = str(scene.get("role") or "").strip().lower()
    section = str(evidence_item.get("section") or "").strip().lower()
    if role == "takeaway" and evidence_item.get("source") == "section_excerpt":
        return False
    allowed_sections = ROLE_EVIDENCE_SECTIONS.get(role)
    if allowed_sections and section not in allowed_sections:
        return False

    claim = str(evidence_item.get("claim") or "")
    if _looks_truncated(claim):
        return False
    scene_tokens = _content_tokens(_scene_text(scene))
    claim_tokens = _content_tokens(claim)
    overlap = scene_tokens & claim_tokens
    if len(overlap) >= 1 and evidence_item.get("source") == "section_excerpt":
        return True
    if len(overlap) >= 2:
        return True

    scene_lower = _scene_text(scene).lower()
    claim_lower = claim.lower()
    role_keyword_pairs = {
        "technical_insight": (("covariance", "uncertainty", "unif", "taxonomy"), ("covariance", "uncertainty", "unif", "covariance-reset")),
        "mechanism": (("covariance", "filter", "uncertainty"), ("covariance", "filter", "uncertainty")),
        "method_overview": (("design", "filter", "framework", "probabilistic"), ("design", "filter", "framework", "probabilistic")),
        "method": (("design", "filter", "framework", "probabilistic"), ("design", "filter", "framework", "probabilistic")),
        "comparison": (("unif", "taxonomy", "ruler", "retrieval", "distill", "probabilistic"), ("unif", "ruler", "retrieval", "distill", "probabilistic")),
        "results": (("ruler", "retrieval", "distill", "benchmark"), ("ruler", "retrieval", "distill", "benchmark")),
        "key_evidence": (("collision", "flood", "extrapolat", "covariance"), ("collision", "flood", "extrapolat", "covariance")),
        "problem": (("opaque", "heuristic", "update", "overwrit", "assumption"), ("obscure", "update", "overwrit", "assumption")),
        "boundary": (("design", "implication", "future", "assumption", "explicit"), ("design", "larger", "space", "assumption", "explicit")),
        "design_principle": (("memory", "design", "principle", "assumption"), ("memory", "design", "assumption", "framework")),
        "takeaway": (("memory", "framework", "uncertainty", "retrieval"), ("memory", "framework", "uncertainty", "retrieval")),
    }
    expected = role_keyword_pairs.get(role)
    if expected and any(word in scene_lower for word in expected[0]) and any(word in claim_lower for word in expected[1]):
        return True

    return role in {"takeaway", "surprising_finding"} and bool(overlap)


def _find_matching_video_evidence(
    scene: dict,
    evidence: list[dict],
    used_keys: set[str],
    allow_used: bool = False,
) -> Optional[dict]:
    for item in evidence:
        if not isinstance(item, dict):
            continue
        key = _evidence_key(item)
        if not allow_used and key in used_keys:
            continue
        if _video_evidence_matches_scene(scene, item):
            return item
    return None


def _find_section_video_evidence(
    scene: dict,
    evidence: list[dict],
    used_keys: set[str],
    allow_used: bool = False,
) -> Optional[dict]:
    role = str(scene.get("role") or "").strip().lower()
    allowed_sections = ROLE_EVIDENCE_SECTIONS.get(role) or set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        key = _evidence_key(item)
        if not allow_used and key in used_keys:
            continue
        claim = str(item.get("claim") or "")
        section = str(item.get("section") or "").strip().lower()
        if _looks_truncated(claim):
            continue
        if allowed_sections and section not in allowed_sections:
            continue
        if role == "takeaway" and item.get("source") == "section_excerpt":
            continue
        return item
    return None


def _clean_video_scene_evidence(scene: dict, evidence: list[dict], used_keys: set[str]) -> None:
    scene_evidence = scene.get("evidence")
    if not isinstance(scene_evidence, dict):
        replacement = (
            _find_matching_video_evidence(scene, evidence, used_keys)
            or _find_matching_video_evidence(scene, evidence, used_keys, allow_used=True)
            or _find_section_video_evidence(scene, evidence, used_keys)
            or _find_section_video_evidence(scene, evidence, used_keys, allow_used=True)
        )
        if replacement:
            scene["evidence"] = replacement
            used_keys.add(_evidence_key(replacement))
        return

    claim = str(scene_evidence.get("claim") or "")
    if _looks_truncated(claim) or not _video_evidence_matches_scene(scene, scene_evidence):
        replacement = (
            _find_matching_video_evidence(scene, evidence, used_keys)
            or _find_matching_video_evidence(scene, evidence, used_keys, allow_used=True)
            or _find_section_video_evidence(scene, evidence, used_keys)
            or _find_section_video_evidence(scene, evidence, used_keys, allow_used=True)
        )
        if replacement:
            scene["evidence"] = replacement
            used_keys.add(_evidence_key(replacement))
        else:
            scene.pop("evidence", None)
        return

    used_keys.add(_evidence_key(scene_evidence))


def _dedupe_video_evidence(scenes: list[dict], evidence: list[dict], max_reuse: int = 2) -> None:
    if not evidence:
        return

    usage: dict[str, int] = {}
    replacement_index = 0
    for scene in scenes:
        key = _evidence_key(scene.get("evidence"))
        if not key:
            continue
        usage[key] = usage.get(key, 0) + 1
        if usage[key] <= max_reuse:
            continue

        for _ in range(len(evidence)):
            replacement = evidence[replacement_index % len(evidence)]
            replacement_index += 1
            replacement_key = _evidence_key(replacement)
            if usage.get(replacement_key, 0) < max_reuse and _video_evidence_matches_scene(scene, replacement):
                scene["evidence"] = replacement
                usage[replacement_key] = usage.get(replacement_key, 0) + 1
                break
        else:
            scene.pop("evidence", None)


def _fill_missing_video_evidence(scenes: list[dict], evidence: list[dict]) -> None:
    if not evidence:
        return

    used_keys = {
        _evidence_key(scene.get("evidence"))
        for scene in scenes
        if _evidence_key(scene.get("evidence"))
    }
    for scene in scenes:
        if isinstance(scene.get("evidence"), dict):
            continue
        replacement = (
            _find_matching_video_evidence(scene, evidence, used_keys)
            or _find_section_video_evidence(scene, evidence, used_keys)
            or _find_matching_video_evidence(scene, evidence, used_keys, allow_used=True)
            or _find_section_video_evidence(scene, evidence, used_keys, allow_used=True)
        )
        if replacement:
            scene["evidence"] = replacement
            used_keys.add(_evidence_key(replacement))


def generate_video_script(analysis_result: dict, slide_count: int = 10):
    slide_count = max(3, min(int(slide_count or 10), 20))
    document_summary = analysis_result.get("document_summary", {})
    paper_title = (
        analysis_result.get("paper_title")
        or document_summary.get("title")
        or analysis_result.get("title")
        or ""
    )
    summary = document_summary.get("summary", "")
    key_ideas = document_summary.get("key_ideas", [])
    contributions = document_summary.get("contributions", [])
    evidence_sources = compact_video_sources(analysis_result.get("evidence_sources", []))
    evidence = rank_video_evidence([
        *(document_summary.get("evidence", []) if isinstance(document_summary.get("evidence"), list) else []),
        *source_video_evidence_candidates(evidence_sources),
    ])
    strongest_evidence = evidence[0] if evidence else {}
    scene_roles = build_video_scene_roles(slide_count)
    slide_profile = video_slide_profile(slide_count)
    duration_range = slide_profile["duration_range"]
    duration_bounds = [int(value) for value in re.findall(r"\d+", duration_range)]
    target_duration = sum(duration_bounds[:2]) // 2 if len(duration_bounds) >= 2 else max(60, slide_count * 15)
    evidence_reuse_limit = max_video_evidence_reuse(slide_count)
    allowed_role_set = set(scene_roles)

    prompt = f"""
    You are converting a research paper analysis into a research presentation deck.
    Only use information from the provided analysis. Do not invent details.

    Create a {duration_range} second {slide_profile["name"]} presentation with exactly {slide_count} slides.
    The output should feel like a conference talk or research group presentation, not like a paper summary divided into pages.

    Required narrative roles, in order:
    {json.dumps(scene_roles, ensure_ascii=False)}

    Slide-count strategy:
    - {slide_profile["structure_rule"]}
    - {slide_profile["evidence_rule"]}

    Planning step:
    - First create deck_plan before writing slides.
    - deck_plan must contain one object per slide with: slide_number, role, question, purpose, takeaway.
    - The plan must show how the story moves forward. Do not create two slides with the same purpose.
    - Use the role sequence above, but adapt the angle to the specific paper.

    Presentation narrative rules:
    - Do not simply summarize the paper slide by slide.
    - Transform the paper into an engaging presentation for the selected audience.
    - Each slide must answer one clear question, such as:
      What is surprising?
      What problem does the paper solve?
      Why should the audience care?
      What is the core technical idea?
      What evidence supports it?
      What are the limitations?
      What should the audience remember?
    - Each slide must move the story forward. Avoid repeating the same idea in different wording.
    - Prefer concrete findings over abstract statements.
    - Avoid generic bullets like "Machine learning is important" or "Further research is needed."
    - Prefer paper-specific claims, for example: "A change in an external API can silently break downstream ML statistics pipelines."
    - Each slide needs one memorable takeaway.
    - The heading should be presentation-style, not section-title style. Avoid headings like "Method", "Results", or "Conclusion" unless paired with a specific claim.
    - Slide 1 must be a strong hook explaining why the paper matters. It should not start with a generic overview.
    - The final slide must be a durable lesson or takeaway, not a recap of previous bullets.

    Slide writing rules:
    - Each slide has 1 strong title.
    - Each slide has 2-3 bullets maximum.
    - Each bullet must be short, concrete, and useful as slide text.
    - Voiceover should explain the slide naturally in presentation language.
    - Do not use the same phrase or claim across multiple slide headings.
    - Do not stretch short decks by adding filler. Short decks should focus on hook → problem → insight → implication → takeaway.
    - Longer decks may include method, mechanism, evidence, limitations, comparison, and future implications, but only when each adds a new narrative beat.

    Evidence rules:
    - At least half of the slides should include an evidence object copied or paraphrased from the Evidence list when suitable support exists.
    - Evidence should ground the slide, but the slide must still read like a presentation, not like a citation list.
    - Evidence may be section-level support; it does not need to prove every word on the slide.
    - If no available evidence reasonably supports a scene, set evidence to null instead of attaching a weak or unrelated claim.
    - Do not reuse the same evidence claim in more than {evidence_reuse_limit} slide(s).
    - Match each slide's evidence broadly to its role: method slides need method/framework evidence; finding/evidence slides need experiment/results evidence; limitation/implication slides need conclusion or boundary evidence.
    - Evidence claims must be complete sentences. Never output a truncated excerpt.
    - Slide 1 or Slide 2 should use the Strongest evidence when it supports the hook or problem.
    - Prefer claims with page numbers when available.

    Safety and specificity:
    - Avoid generic hype like "transforming every industry" unless directly supported.
    - Do not turn a listed example into a causal claim.
    - If the paper lists or compares models, say "the paper compares/lists models such as..." not "the shift is driven by..."
    - Do not claim a model, method, or factor drives a field-wide change unless the evidence explicitly says so.
    - Prefer careful wording such as "the paper compares", "the paper classifies", "can be fine-tuned", or "open-source models allow researchers to adapt..."
    - Avoid consultant-style endings such as "monitor, mitigate, innovate".
    - If a paper has no explicit risk, limitation, or future-work evidence, use limitation/implication to describe a supported boundary or design implication instead of inventing a risk.

    Visual rules:
    - visual_note must describe a text-renderable slide layout only: title emphasis, short bullets, simple comparison labels, or a clean takeaway card.
    - Do not request graphs, charts, plots, tables, Venn diagrams, icons, animations, or illustrations unless the information can be represented as plain text bullets.

    Use the provided paper title as the script title when available.

    Return ONLY valid JSON:
    {{
      "title": "...",
      "duration_seconds": {target_duration},
      "audience": "software engineers and research readers",
      "deck_plan": [
        {{
          "slide_number": 1,
          "role": "hook",
          "question": "What is surprising?",
          "purpose": "...",
          "takeaway": "..."
        }}
      ],
      "scenes": [
        {{
          "scene_number": 1,
          "role": "hook",
          "heading": "...",
          "bullets": ["...", "..."],
          "voiceover": "...",
          "evidence": {{
            "claim": "...",
            "section": "abstract|introduction|method|experiment|results|conclusion|related_work",
            "pages": [1, 2]
          }},
          "visual_type": "comparison|evidence_card|classification|boundary|takeaway",
          "visual_note": "Describe one simple Apple keynote-style text layout that can be rendered without charts or images."
        }}
      ]
    }}

    Analysis:
    Paper title:
    {paper_title}

    Summary:
    {summary}

    Key ideas:
    {json.dumps(key_ideas, ensure_ascii=False)}

    Contributions:
    {json.dumps(contributions, ensure_ascii=False)}

    Strongest evidence:
    {json.dumps(strongest_evidence, ensure_ascii=False)}

    Evidence:
    {json.dumps(evidence, ensure_ascii=False)}

    Source sections:
    {json.dumps(evidence_sources, ensure_ascii=False)}
    """

    try:
        script = generate_json(prompt)
        scenes = script.get("scenes", [])
        if isinstance(scenes, list):
            cleaned_scenes = [scene for scene in scenes if isinstance(scene, dict)]
            if len(cleaned_scenes) > slide_count:
                cleaned_scenes = cleaned_scenes[:slide_count]
            while len(cleaned_scenes) < slide_count:
                role = scene_roles[min(len(cleaned_scenes), len(scene_roles) - 1)]
                cleaned_scenes.append(
                    make_fallback_video_scene(len(cleaned_scenes) + 1, role, analysis_result, evidence)
                )
            script["scenes"] = cleaned_scenes
            default_roles = scene_roles
            used_evidence_keys: set[str] = set()
            for index, scene in enumerate(cleaned_scenes, start=1):
                scene["scene_number"] = index
                role = str(scene.get("role") or "").strip()
                if role not in allowed_role_set:
                    scene["role"] = default_roles[min(index - 1, len(default_roles) - 1)]
                if index in {1, 2} and strongest_evidence:
                    scene.setdefault("evidence", strongest_evidence)
                _clean_video_scene_evidence(scene, evidence, used_evidence_keys)
                scene.setdefault("visual_type", "evidence_card" if scene.get("evidence") else "takeaway")
                normalize_video_scene(scene)
            _dedupe_video_evidence(cleaned_scenes, evidence, evidence_reuse_limit)
            _fill_missing_video_evidence(cleaned_scenes, evidence)
            script["slide_count"] = len(cleaned_scenes)
            script.setdefault("duration_seconds", target_duration)
        deck_plan = script.get("deck_plan")
        if isinstance(deck_plan, list):
            cleaned_plan = [item for item in deck_plan if isinstance(item, dict)][:slide_count]
            for index, item in enumerate(cleaned_plan, start=1):
                item["slide_number"] = index
                role = str(item.get("role") or "").strip()
                if role not in allowed_role_set:
                    item["role"] = scene_roles[min(index - 1, len(scene_roles) - 1)]
            script["deck_plan"] = cleaned_plan
        if paper_title:
            script["paper_title"] = paper_title
            script["title"] = paper_title
        return script
    except json.JSONDecodeError as error:
        return {
            "title": "Video script generation failed",
            "duration_seconds": 0,
            "audience": "",
            "scenes": [],
            "error": error.doc,
        }

def extract_references_llm(ref_text: str):
    prompt = f"""
    You are extracting the bibliography from a research paper.
    Only use information from the provided text. Do not invent details.

    Extract complete reference entries from the text.
    Keep each reference as one string.
    Do not summarize, rewrite, or add missing information.
    Return as many complete reference entries as are present in the provided text, up to 80 references.
    Exclude incomplete or truncated references.

    Return ONLY valid JSON:
    {{
      "references": ["...", "..."]
    }}

    Text:
    {ref_text}
    """

    try:
        return generate_json(prompt).get("references", [])
    except json.JSONDecodeError:
        return []
        
