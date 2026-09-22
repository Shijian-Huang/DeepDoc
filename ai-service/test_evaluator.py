import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm.evaluator import EvidencePacket, EvaluationError, SummaryEvaluator, _extract_json
from utils.summary_prompt import build_research_summary_prompt


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
            "total_summary_claims": 7,
        }


class SummaryEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.packet = EvidencePacket("paper-section", "Ground-truth paper excerpt.")

    def test_extract_json_ignores_trailing_model_output(self):
        self.assertEqual(
            json.loads(_extract_json('{"status": "ok"}\n{"extra": true}')),
            {"status": "ok"},
        )

    def test_default_mode_calculates_scores_for_each_model_and_packet(self):
        evaluator = SummaryEvaluator(
            {"model-a": "Summary A", "model-b": {"summary": "Summary B"}},
            [self.packet, {"id": "second", "text": "Another excerpt."}],
            judge=FakeJudge(),
        )

        result = evaluator.evaluate()

        self.assertEqual(list(result), ["model-a", "model-b"])
        self.assertEqual(len(result["model-a"]), 2)
        self.assertEqual(len(result["model-b"]), 2)
        self.assertEqual(result["model-a"][0], {
            "packet_id": "paper-section",
            "evaluation": {
                "avg_score": 0.5196,
                "key_ideas_score": 0.7,
                "contributions_score": 0.625,
                "hallucination_score": 0.1429,
            },
        })
        json.dumps(result, allow_nan=False)

    def test_detailed_mode_has_requested_public_shape(self):
        result = SummaryEvaluator(
            {"model-a": "Summary"},
            [self.packet],
            judge=FakeJudge(),
        ).evaluate("detailed")["model-a"][0]["evaluation"]

        self.assertEqual(result["key_ideas"], {
            "covered": 3,
            "expected": 5,
            "missing": ["One missing idea"],
        })
        self.assertEqual(result["contributions"], {"covered": 2, "missing": 1})
        self.assertEqual(len(result["hallucinated_claims"]), 1)

    def test_model_names_use_injected_summary_generator(self):
        calls = []

        def generate(model, prompt):
            calls.append((model, prompt))
            return {"summary": f"Summary from {model}"}

        evaluator = SummaryEvaluator(
            ["local-model", "remote-model"],
            [self.packet],
            judge=FakeJudge(),
            summary_generator=generate,
        )
        evaluator.evaluate()

        expected_prompt = build_research_summary_prompt(self.packet.text, "standard")
        self.assertEqual(calls, [
            ("local-model", expected_prompt),
            ("remote-model", expected_prompt),
        ])

    def test_model_names_without_generator_are_rejected(self):
        with self.assertRaises(EvaluationError):
            SummaryEvaluator("model-a", [self.packet], judge=FakeJudge())

    def test_default_packets_are_used_when_packets_are_omitted(self):
        evaluator = SummaryEvaluator({"model-a": "Summary"}, judge=FakeJudge())
        result = evaluator.evaluate()
        self.assertGreaterEqual(len(result["model-a"]), 1)

    def test_multiple_callable_models_are_supported(self):
        prompts = []

        def model_a(prompt):
            prompts.append(prompt)
            return {"summary": "Summary A"}

        def model_b(prompt):
            prompts.append(prompt)
            return {"summary": "Summary B"}

        evaluator = SummaryEvaluator(
            {"model-a": model_a, "model-b": model_b},
            [self.packet],
            judge=FakeJudge(),
        )
        result = evaluator.evaluate()
        self.assertEqual(list(result), ["model-a", "model-b"])
        self.assertEqual(len(result["model-a"]), 1)
        self.assertEqual(len(result["model-b"]), 1)
        self.assertEqual(prompts[0], prompts[1])
        self.assertIn(self.packet.text, prompts[0])

    def test_average_mode_averages_packet_scores_per_model(self):
        class PacketJudge:
            def evaluate(inner_self, summary, evidence_packet):
                covered = 5 if evidence_packet.id == "complete" else 0
                contribution_covered = 4 if evidence_packet.id == "complete" else 0
                hallucinations = [] if evidence_packet.id == "complete" else [
                    {"claim": "Unsupported", "reason": "Absent", "evidence": None}
                ]
                return {
                    "key_ideas": {
                        "covered": covered,
                        "partial_covered": 0,
                        "expected": 5,
                        "missing": [] if covered else ["a", "b", "c", "d", "e"],
                    },
                    "contributions": {
                        "covered": contribution_covered,
                        "partial_covered": 0,
                        "expected": 4,
                        "missing": [] if contribution_covered else ["a", "b", "c", "d"],
                    },
                    "hallucinated_claims": hallucinations,
                    "total_summary_claims": 9 if covered else 1,
                }

        result = SummaryEvaluator(
            {"model-a": "Summary", "model-b": "Another summary"},
            [
                {"id": "complete", "text": "First excerpt."},
                {"id": "empty", "text": "Second excerpt."},
            ],
            judge=PacketJudge(),
        ).evaluate("average")

        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0], {
            "model": "model-a",
            "packet_count": 2,
            "evaluation": {
                "avg_score": 0.5,
                "key_ideas_score": 0.5,
                "contributions_score": 0.5,
                "hallucination_score": 0.5,
            },
        })

    def test_all_hallucinated_claims_are_fully_penalized(self):
        class HallucinationJudge:
            def evaluate(inner_self, summary, evidence_packet):
                return {
                    "key_ideas": {"covered": 0, "partial_covered": 0, "expected": 1, "missing": ["idea"]},
                    "contributions": {"covered": 0, "partial_covered": 0, "expected": 1, "missing": ["contribution"]},
                    "hallucinated_claims": [{"claim": "Unsupported", "reason": "Absent", "evidence": None}],
                    "total_summary_claims": 1,
                }

        evaluation = SummaryEvaluator(
            {"model-a": "Unsupported summary"},
            [self.packet],
            judge=HallucinationJudge(),
        ).evaluate()["model-a"][0]["evaluation"]

        self.assertEqual(evaluation["hallucination_score"], 1.0)
        self.assertEqual(evaluation["avg_score"], 0.0)

    def test_average_mode_saves_visualization_in_evaluation_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = SummaryEvaluator(
                {"model-a": "Summary", "model-b": "Another summary"},
                [self.packet],
                judge=FakeJudge(),
                evaluation_dir=temp_dir,
            ).evaluate("average", visualize=True)

            chart_path = Path(result["visualization_path"])
            self.assertEqual(chart_path.parent, Path(temp_dir).resolve())
            self.assertEqual(chart_path.suffix, ".png")
            self.assertTrue(chart_path.is_file())
            self.assertEqual(chart_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertNotIn("visualization", result)
            json.dumps(result, allow_nan=False)

    def test_visualization_is_rejected_outside_average_mode(self):
        evaluator = SummaryEvaluator(
            {"model-a": "Summary"},
            [self.packet],
            judge=FakeJudge(),
        )
        with self.assertRaises(EvaluationError):
            evaluator.evaluate("default", visualize=True)


if __name__ == "__main__":
    unittest.main()
