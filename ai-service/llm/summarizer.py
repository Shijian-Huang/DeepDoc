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
        
