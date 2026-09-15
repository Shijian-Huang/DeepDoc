import json
import logging
import threading

from llm.summarizer import generate_json

logger = logging.getLogger(__name__)

_evaluation_results: dict[str, dict] = {}
_results_lock = threading.Lock()
MAX_STORED_RESULTS = 100


def _store_result(analysis_id: str, result: dict) -> None:
    with _results_lock:
        if len(_evaluation_results) >= MAX_STORED_RESULTS:
            oldest_key = next(iter(_evaluation_results))
            del _evaluation_results[oldest_key]
        _evaluation_results[analysis_id] = result


def get_evaluation(analysis_id: str) -> dict | None:
    with _results_lock:
        return _evaluation_results.get(analysis_id)


def build_evaluation_prompt(document_summary: dict, evidence_packet: str) -> str:
    summary_text = document_summary.get("summary", "")
    key_ideas = document_summary.get("key_ideas", [])
    contributions = document_summary.get("contributions", [])
    evidence = document_summary.get("evidence", [])

    return f"""
    You are a research paper summary evaluator.
    You will be given the original paper text and a generated summary.
    Your job is to evaluate the summary on three dimensions:

    1. Key idea coverage:
       Read the paper and identify its key ideas (main concepts, methods, findings).
       Then check which of those key ideas appear in the generated summary.
       A key idea is "covered" if the summary captures it accurately.
       A key idea is "partial_covered" if mentioned but missing important nuance or detail.
       A key idea is "missing" if not mentioned at all.

    2. Contribution coverage:
       Read the paper and identify the contributions it claims (novel methods, results, insights).
       Then check which contributions appear in the generated summary.
       Apply the same covered / partial_covered / missing criteria.

    3. Hallucination detection:
       Go through every factual claim in the summary text, key ideas, and contributions.
       A claim is hallucinated if it cannot be traced to information in the paper text.
       Be strict: vague restatements are acceptable, but invented facts, numbers, or
       relationships not in the paper are hallucinations.
       Count the total number of distinct factual claims in the summary.

    4. Coherence:
       Rate how well the summary reads as a logically structured whole on a 1-5 scale.
       Consider: Are paragraphs ordered sensibly? Do ideas flow with clear transitions?
       Are there abrupt topic jumps or disjointed segments?
       1 = incoherent / random ordering, 5 = perfectly structured and flowing.

    5. Consistency:
       Rate how internally consistent the summary is on a 1-5 scale.
       Consider: Does the summary contradict itself? Are claims, terminology, and framing
       uniform throughout? Does it say one thing in one paragraph and something different
       in another?
       1 = frequent contradictions, 5 = fully consistent throughout.

    Return ONLY valid JSON:
    {{
      "key_ideas": {{
        "covered": 0,
        "partial_covered": 0,
        "expected": 0,
        "missing": ["..."]
      }},
      "contributions": {{
        "covered": 0,
        "partial_covered": 0,
        "expected": 0,
        "missing": ["..."]
      }},
      "hallucinated_claims": [
        {{
          "claim": "the exact claim from the summary",
          "reason": "why it is not supported",
          "evidence": null
        }}
      ],
      "total_summary_claims": 0,
      "coherence": {{
        "score": 0,
        "justification": "..."
      }},
      "consistency": {{
        "score": 0,
        "justification": "..."
      }}
    }}

    --- PAPER TEXT ---
    {evidence_packet}

    --- GENERATED SUMMARY ---
    Summary:
    {summary_text}

    Key Ideas:
    {json.dumps(key_ideas, ensure_ascii=False)}

    Contributions:
    {json.dumps(contributions, ensure_ascii=False)}

    Evidence:
    {json.dumps(evidence, ensure_ascii=False)}
    """


def calculate_evaluation_scores(evaluation: dict) -> dict:
    key_ideas = evaluation.get("key_ideas", {})
    contributions = evaluation.get("contributions", {})
    hallucinated_claims = evaluation.get("hallucinated_claims", [])
    total_claims = evaluation.get("total_summary_claims", 0)

    ki_expected = key_ideas.get("expected", 0)
    ki_covered = key_ideas.get("covered", 0)
    ki_partial = key_ideas.get("partial_covered", 0)

    co_expected = contributions.get("expected", 0)
    co_covered = contributions.get("covered", 0)
    co_partial = contributions.get("partial_covered", 0)

    key_ideas_score = (ki_covered + 0.5 * ki_partial) / ki_expected if ki_expected > 0 else 0.0
    contributions_score = (co_covered + 0.5 * co_partial) / co_expected if co_expected > 0 else 0.0

    hallucination_count = len(hallucinated_claims) if isinstance(hallucinated_claims, list) else 0
    hallucination_rate = hallucination_count / total_claims if total_claims > 0 else 0.0

    coherence_raw = evaluation.get("coherence", {})
    consistency_raw = evaluation.get("consistency", {})
    coherence_score = max(0.0, min(1.0, (coherence_raw.get("score", 0) - 1) / 4)) if isinstance(coherence_raw, dict) else 0.0
    consistency_score = max(0.0, min(1.0, (consistency_raw.get("score", 0) - 1) / 4)) if isinstance(consistency_raw, dict) else 0.0

    return {
        "key_ideas_coverage": round(key_ideas_score, 4),
        "contributions_coverage": round(contributions_score, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "coherence": round(coherence_score, 4),
        "consistency": round(consistency_score, 4),
    }


def evaluate_summary(document_summary: dict, evidence_packet: str) -> dict:
    prompt = build_evaluation_prompt(document_summary, evidence_packet)

    try:
        evaluation = generate_json(prompt)
        scores = calculate_evaluation_scores(evaluation)
        return {
            "evaluation": evaluation,
            "scores": scores,
        }
    except json.JSONDecodeError as error:
        return {
            "evaluation": {},
            "scores": {},
            "error": error.doc,
        }


def run_evaluation(
    document_summary: dict,
    evidence_packet: str,
    summary_mode: str,
    analysis_id: str | None = None,
) -> None:
    logger.info("Starting summary evaluation (mode: %s)...", summary_mode)
    result = evaluate_summary(document_summary, evidence_packet)

    if "error" in result:
        logger.error("Summary evaluation failed: %s", result["error"])
        if analysis_id:
            _store_result(analysis_id, result)
        return

    logger.info(
        "Summary evaluation complete.\n%s",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    if analysis_id:
        _store_result(analysis_id, result)
