"""Shared research-summary prompt construction.

Keeping this module independent from a model provider ensures every summary
model can receive the same prompt and evidence packet.
"""

from typing import Optional


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

SUMMARY_MODE_TARGET_WORDS = {
    "paragraph": "100-150",
    "standard": "220-300",
    "one_page": "450-700",
}


def normalize_summary_mode(summary_mode: str) -> str:
    if summary_mode in SUMMARY_MODE_INSTRUCTIONS:
        return summary_mode
    return "standard"


def build_research_summary_prompt(
    evidence_packet: str,
    summary_mode: str,
    retry_word_count: Optional[int] = None,
) -> str:
    normalized_mode = normalize_summary_mode(summary_mode)
    length_instruction = SUMMARY_MODE_INSTRUCTIONS[normalized_mode]
    retry_instruction = ""
    if retry_word_count is not None:
        retry_instruction = f"""
    Previous attempt was too short at {retry_word_count} words.
    Rewrite and expand the summary field to {SUMMARY_MODE_TARGET_WORDS[normalized_mode]} words.
    Keep the summary faithful to the provided paper text.
    """

    return f"""
    You are analyzing a research paper from selected high-value sections.
    Only use information from the provided text. Do not invent details.

    Summary mode: {normalized_mode}
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
