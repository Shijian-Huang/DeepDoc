import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm.evaluator import EvidencePacket, EvaluationError, SummaryEvaluator


class FakeJudge:
    def evaluate(self, summary, evidence_packet):
        self.last_summary = summary
        self.last_packet = evidence_packet
        return {
            "key_ideas": {
                "covered": 3,
                "partial_covered": 1,
                "expected": 5,
                "missing": ["One missing idea"],
            },
            "contributions": {
                "covered": 2,
                "partial_covered": 1,
                "expected": 4,
                "missing": ["One missing contribution"],
            },
            "hallucinated_claims": [
                {"claim": "Unsupported claim", "reason": "Not in packet", "evidence": None}
            ],
        }


class SummaryEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.packet = EvidencePacket("paper-section", "Ground-truth paper excerpt.")

    def test_default_mode_calculates_scores_for_each_model_and_packet(self):
        evaluator = SummaryEvaluator(
            {"model-a": "Summary A", "model-b": {"summary": "Summary B"}},
            [self.packet, {"id": "second", "text": "Another excerpt."}],
            judge=FakeJudge(),
        )

        result = evaluator.evaluate()

        self.assertEqual(len(result["results"]), 4)
        self.assertEqual(result["results"][0]["evaluation"], {
            "avg_score": 0.6625,
            "key_ideas_score": 0.7,
            "contributions_score": 0.625,
            "hallucination_score": 0.1,
        })
        json.dumps(result, allow_nan=False)

    def test_detailed_mode_has_requested_public_shape(self):
        result = SummaryEvaluator(
            {"model-a": "Summary"},
            [self.packet],
            judge=FakeJudge(),
        ).evaluate("detailed")["results"][0]["evaluation"]

        self.assertEqual(result["key_ideas"], {
            "covered": 3,
            "expected": 5,
            "missing": ["One missing idea"],
        })
        self.assertEqual(result["contributions"], {"covered": 2, "missing": 1})
        self.assertEqual(len(result["hallucinated_claims"]), 1)

    def test_model_names_use_injected_summary_generator(self):
        calls = []

        def generate(model, packet):
            calls.append((model, packet.id))
            return {"summary": f"Summary from {model}"}

        evaluator = SummaryEvaluator(
            ["local-model", "remote-model"],
            [self.packet],
            judge=FakeJudge(),
            summary_generator=generate,
        )
        evaluator.evaluate()

        self.assertEqual(calls, [
            ("local-model", "paper-section"),
            ("remote-model", "paper-section"),
        ])

    def test_model_names_without_generator_are_rejected(self):
        with self.assertRaises(EvaluationError):
            SummaryEvaluator("model-a", [self.packet], judge=FakeJudge())

    def test_default_packets_are_used_when_packets_are_omitted(self):
        evaluator = SummaryEvaluator({"model-a": "Summary"}, judge=FakeJudge())
        result = evaluator.evaluate()
        self.assertGreaterEqual(len(result["results"]), 1)
    def test_multiple_callable_models_are_supported(self):
        def model_a(packet):
            return {"summary": "Summary A"}

        def model_b(packet):
            return {"summary": "Summary B"}

        evaluator = SummaryEvaluator(
            {"model-a": model_a, "model-b": model_b},
            [self.packet],
            judge=FakeJudge(),
        )
        result = evaluator.evaluate()
        self.assertEqual(len(result["results"]), 2)


if __name__ == "__main__":
    unittest.main()
