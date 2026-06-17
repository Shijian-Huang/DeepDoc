import os
import json
import re
import time
from pathlib import Path
from typing import Optional
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

SUMMARY_MODE_INSTRUCTIONS = {
    "paragraph": (
        "Write exactly one concise paragraph. The summary field must be 80-120 words."
    ),
    "standard": (
        "Write 2-3 developed paragraphs. The summary field must be 220-300 words. "
        "Do not write fewer than 200 words."
    ),
    "one_page": (
        "Write 4-6 well-developed paragraphs. The summary field must be 500-650 words. "
        "Do not write fewer than 450 words."
    ),
}

SUMMARY_MODE_MIN_WORDS = {
    "paragraph": 80,
    "standard": 200,
    "one_page": 450,
}

SUMMARY_MODE_TARGET_WORDS = {
    "paragraph": "80-120",
    "standard": "220-300",
    "one_page": "500-650",
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


def needs_summary_expansion(result: dict, summary_mode: str) -> bool:
    summary = result.get("summary", "")
    return count_words(summary) < SUMMARY_MODE_MIN_WORDS[summary_mode]


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
    The length requirement applies only to the "summary" field.
    Do not count key_ideas, contributions, references, evidence, or summary_word_count toward the word count.
    {retry_instruction}

    Focus on:
    - Main problem
    - Core method
    - Key findings
    - Main contributions
    - Evidence-grounded claims

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
        word_count = count_words(result.get("summary", ""))
        result["summary_word_count"] = word_count

        if needs_summary_expansion(result, normalized_mode):
            retry_prompt = build_research_summary_prompt(
                evidence_packet,
                normalized_mode,
                retry_word_count=word_count,
            )
            retry_result = generate_json(retry_prompt)
            retry_result["summary_word_count"] = count_words(
                retry_result.get("summary", "")
            )
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


def compact_video_sources(sources: list, limit: int = 4, excerpt_chars: int = 260) -> list[dict]:
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


def generate_video_script(analysis_result: dict):
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
    evidence = rank_video_evidence(document_summary.get("evidence", []))
    evidence_sources = compact_video_sources(analysis_result.get("evidence_sources", []))
    strongest_evidence = evidence[0] if evidence else {}

    prompt = f"""
    You are converting a research paper analysis into a short research explainer video script.
    Only use information from the provided analysis. Do not invent details.

    Create a 60-90 second video with exactly 5 scenes.
    Do not simply restate the summary. The video should answer:
    "What are the 3-5 things a viewer should remember from this paper?"

    First plan the video with these fixed scene roles, then fill each scene:
    1. surprising_finding: the most surprising paper-backed result. Use Strongest evidence when available.
    2. why_it_matters: explain the implication of that result for the target audience.
    3. technical_insight: explain a mechanism, classification, model property, or concrete example from the paper.
    4. risk: explain the main tension, limitation, or dual-use concern.
    5. takeaway: close with the durable lesson. Do not introduce a new topic in the final scene.

    Requirements:
    - Each scene must have one sharp takeaway, not a mini-summary.
    - At least 3 scenes must include an evidence object copied or paraphrased from the Evidence list.
    - Scene 1 or Scene 2 must use the Strongest evidence unless it is empty.
    - If the strongest evidence says LLMs outperform commercial tools, make that the hook.
    - Scene 1 heading should sound like a surprising result, not a broad topic. For example: "The Surprising Result", "The Result That Matters", or a short domain-specific version of the strongest evidence.
    - Prefer claims with page numbers when available.
    - Bullets must be short slide text, 3-8 words each.
    - Voiceover should be natural, explanatory, and specific to this paper.
    - Avoid generic hype like "transforming every industry" unless directly supported.
    - Do not turn a listed example into a causal claim. If the paper lists or compares models, say "the paper compares/lists models such as..." not "the shift is driven by..."
    - Do not claim a model, method, or factor drives a field-wide change unless the evidence explicitly says so.
    - For technical_insight, prefer careful wording such as "the paper compares", "the paper classifies", "can be fine-tuned", or "open-source models allow researchers to adapt..."
    - Avoid consultant-style endings such as "monitor, mitigate, innovate".
    - The final scene should synthesize earlier scenes: LLMs are becoming both security tools and security risks; the question is how to use them safely.
    Use the provided paper title as the script title when available.

    Return ONLY valid JSON:
    {{
      "title": "...",
      "duration_seconds": 80,
      "audience": "software engineers and research readers",
      "scenes": [
        {{
          "scene_number": 1,
          "role": "surprising_finding|why_it_matters|technical_insight|risk|takeaway",
          "heading": "...",
          "bullets": ["...", "..."],
          "voiceover": "...",
          "evidence": {{
            "claim": "...",
            "section": "abstract|introduction|method|experiment|results|conclusion|related_work",
            "pages": [1, 2]
          }},
          "visual_type": "comparison|evidence_card|classification|risk_map|takeaway",
          "visual_note": "Describe a simple template graphic that can be rendered from text, not an AI image prompt."
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
            script["scenes"] = cleaned_scenes
            default_roles = ["surprising_finding", "why_it_matters", "technical_insight", "risk", "takeaway"]
            for index, scene in enumerate(cleaned_scenes, start=1):
                scene.setdefault("scene_number", index)
                scene.setdefault("role", default_roles[min(index - 1, 4)])
                if not scene.get("evidence") and evidence:
                    scene["evidence"] = evidence[(index - 1) % len(evidence)]
                if index in {1, 2} and strongest_evidence:
                    scene.setdefault("evidence", strongest_evidence)
                scene.setdefault("visual_type", "evidence_card" if scene.get("evidence") else "takeaway")
                normalize_video_scene(scene)
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
    Return at most 10 references.
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
        
