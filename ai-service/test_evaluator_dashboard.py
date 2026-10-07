import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from evaluation.dashboard import DashboardEvaluator, JsonResultStore, build_custom_packet
from llm.evaluator import EvidencePacket


@dataclass
class FakeModel:
    name: str
    calls: int = 0

    def summarize(self, prompt):
        self.calls += 1
        return {"summary": f"{self.name}: {prompt[:20]}"}


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def evaluate(self, summary, evidence_packet):
        self.calls += 1
        covered = 2 if evidence_packet.id == "paper-a" else 1
        return {
            "key_ideas": {
                "covered": covered,
                "partial_covered": 0,
                "expected": 2,
                "missing": [] if covered == 2 else ["idea"],
            },
            "contributions": {
                "covered": covered,
                "partial_covered": 0,
                "expected": 2,
                "missing": [] if covered == 2 else ["contribution"],
            },
            "hallucinated_claims": [],
            "total_summary_claims": 2,
        }


@dataclass
class FailingModel:
    name: str

    def summarize(self, _prompt):
        raise ValueError("invalid model output")


class DashboardEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model_a = FakeModel("Model A")
        self.model_b = FakeModel("Model B")
        self.judge = FakeJudge()
        self.dashboard = DashboardEvaluator(
            models={"Model A": self.model_a, "Model B": self.model_b},
            packets=[
                EvidencePacket("paper-a", "Complete evidence."),
                EvidencePacket("paper-b", "Partial evidence."),
            ],
            judge=self.judge,
            store=JsonResultStore(Path(self.temporary.name) / "cache.json"),
        )

    def test_requested_pairs_are_lazy_and_cached(self):
        first = self.dashboard.evaluate(["Model A"], ["paper-a"], "standard")
        second = self.dashboard.evaluate(["Model A"], ["paper-a"], "standard")

        self.assertFalse(first["results"][0]["cached"])
        self.assertTrue(second["results"][0]["cached"])
        self.assertEqual(self.model_a.calls, 1)
        self.assertEqual(self.model_b.calls, 0)
        self.assertEqual(self.judge.calls, 1)

    def test_force_runs_summary_and_evaluation_again(self):
        self.dashboard.evaluate(["Model A"], ["paper-a"], "paragraph")
        result = self.dashboard.evaluate(
            ["Model A"], ["paper-a"], "paragraph", force=True
        )

        self.assertFalse(result["results"][0]["cached"])
        self.assertEqual(self.model_a.calls, 2)
        self.assertEqual(self.judge.calls, 2)

    def test_all_papers_are_averaged_per_model(self):
        result = self.dashboard.evaluate(
            ["Model A", "Model B"], None, "one_page"
        )

        self.assertTrue(result["averaged"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["paper_count"], 2)
        self.assertEqual(result["results"][0]["evaluation"]["avg_score"], 0.75)
        self.assertEqual(len(result["packet_results"]), 4)

    def test_selected_models_can_be_evaluated_on_one_paper(self):
        result = self.dashboard.evaluate(
            ["Model A", "Model B"], ["paper-a"], "standard"
        )

        self.assertFalse(result["averaged"])
        self.assertFalse(result["partial"])
        self.assertEqual(
            [row["model"] for row in result["results"]],
            ["Model A", "Model B"],
        )
        self.assertTrue(all("evaluation" in row for row in result["results"]))

    def test_custom_summary_can_be_compared_with_models(self):
        packet = EvidencePacket("custom:test", "Uploaded evidence.")
        result = self.dashboard.evaluate_custom(
            "A user-written summary.",
            packet,
            "standard",
            ["Model B"],
        )

        self.assertEqual([row["model"] for row in result["results"]], [
            "Custom summary",
            "Model B",
        ])
        self.assertEqual(self.model_b.calls, 1)

    def test_one_model_failure_does_not_abort_multi_model_comparison(self):
        self.dashboard.models = {
            "Model A": self.model_a,
            "Broken Model": FailingModel("Broken Model"),
        }

        result = self.dashboard.evaluate(
            ["Model A", "Broken Model"], ["paper-a"], "standard"
        )

        self.assertTrue(result["partial"])
        self.assertEqual(result["error_count"], 1)
        self.assertIn("evaluation", result["results"][0])
        self.assertEqual(result["results"][1]["model"], "Broken Model")
        self.assertEqual(result["results"][1]["error"], "invalid model output")

    def test_single_model_failure_is_still_returned_as_request_error(self):
        self.dashboard.models = {"Broken Model": FailingModel("Broken Model")}

        with self.assertRaisesRegex(ValueError, "invalid model output"):
            self.dashboard.evaluate(
                ["Broken Model"], ["paper-a"], "standard"
            )

    def test_text_upload_is_processed_into_an_evidence_packet(self):
        path = Path(self.temporary.name) / "reference.txt"
        path.write_text(
            "Abstract\n" + ("Grounded research evidence and findings. " * 30),
            encoding="utf-8",
        )

        packet = build_custom_packet(path, "standard", "reference.txt")

        self.assertTrue(packet.id.startswith("custom:reference.txt:"))
        self.assertIn("Grounded research evidence", packet.text)

    def test_unknown_mode_model_and_paper_are_rejected(self):
        with self.assertRaises(ValueError):
            self.dashboard.evaluate(["Unknown"], ["paper-a"], "standard")
        with self.assertRaises(ValueError):
            self.dashboard.evaluate(["Model A"], ["unknown"], "standard")
        with self.assertRaises(ValueError):
            self.dashboard.evaluate(["Model A"], ["paper-a"], "invalid")


if __name__ == "__main__":
    unittest.main()
