#!/usr/bin/env python3
"""Exercise SummaryEvaluator with the Gemini summarizer and configured judge.

Pass an evidence-packet JSON file as the optional positional argument. When it
is omitted, the evaluator uses ``evaluation/default_packets.json``.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path
from typing import Any


AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_DIR))

from llm import summarizer  # noqa: E402
from llm.evaluator import (  # noqa: E402
    DEFAULT_PACKETS_PATH,
    EvaluationError,
    create_evaluation_judge,
    SummaryEvaluator,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate summaries with Gemini and evaluate them with the "
            "configured Ollama or Gemini SummaryEvaluator judge."
        ),
    )
    parser.add_argument(
        "packet_file",
        nargs="?",
        type=Path,
        default=DEFAULT_PACKETS_PATH,
        help="JSON evidence-packet file (default: evaluation/default_packets.json).",
    )
    parser.add_argument(
        "--mode",
        choices=("default", "detailed", "average"),
        default="default",
        help="Evaluator output mode (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-mode",
        choices=("paragraph", "standard", "one_page"),
        default="standard",
        help="Prompt format used to generate each summary (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-model",
        help=(
            "Gemini model used for summaries. When omitted, the first model "
            "in the summarizer's configured Gemini model list is used."
        ),
    )
    parser.add_argument(
        "--judge-model",
        help=(
            "Override the judge model. Uses Ollama when OLLAMA_JUDGE_MODEL is set; "
            "otherwise uses GEMINI_EVALUATOR_MODEL or the built-in Gemini default."
        ),
    )
    parser.add_argument(
        "--minicheck-device",
        choices=("cpu", "auto"),
        default="cpu",
        help="MiniCheck device selection; CPU avoids native CUDA initialization crashes (default: %(default)s).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a score chart (only valid with --mode average).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON result to this path instead of stdout.",
    )
    args = parser.parse_args(argv)
    if args.visualize and args.mode != "average":
        parser.error("--visualize requires --mode average")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    # MiniCheck uses device_map="auto" internally. Hide GPUs before its lazy
    # torch import so a broken CUDA driver cannot terminate the evaluator.
    if args.minicheck_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    summarizer.require_gemini_api_key()

    summary_model = args.summary_model or summarizer.gemini_models[0]
    summary_label = summary_model

    def generate_summary(_model_name: str, prompt: str) -> dict[str, Any]:
        with summarizer.use_llm_selection("gemini", summary_model):
            return summarizer.generate_json(prompt)

    evaluator = SummaryEvaluator(
        models=summary_label,
        summary_generator=generate_summary,
        summary_mode=args.summary_mode,
        default_packets_path=args.packet_file,
        judge=create_evaluation_judge(model=args.judge_model),
    )
    return evaluator.evaluate(args.mode, visualize=args.visualize)


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    args = parse_args(argv)
    try:
        result = run(args)
    except (EvaluationError, RuntimeError, OSError, json.JSONDecodeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1

    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(f"Evaluation result: {output_path}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
