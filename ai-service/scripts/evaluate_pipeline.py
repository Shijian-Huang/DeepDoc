#!/usr/bin/env python3
"""Evaluate saved pipeline output against its evidence-source excerpts.

Usage: python scripts/evaluate_pipeline.py --input run.json --output evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline key_ideas, contributions, evidence, and summary against evidence_sources.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input pipeline JSON file.")
    parser.add_argument("--output", required=True, type=Path, help="Output evaluation JSON file.")
    parser.add_argument(
        "--judge-model",
        help="Override the configured Ollama or Gemini judge model.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from llm.evaluator import create_evaluation_judge, evaluate_pipeline_output

    payload = json.loads(args.input.expanduser().read_text(encoding="utf-8"))
    return evaluate_pipeline_output(payload, judge=create_evaluation_judge(model=args.judge_model))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Evaluation result: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
