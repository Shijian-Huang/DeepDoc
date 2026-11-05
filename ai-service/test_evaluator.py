import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm.evaluator import (
    EvidencePacket,
    EvaluationError,
    GeminiJudge,
    OllamaJudge,
    JUDGE_RESPONSE_SCHEMA,
    create_evaluation_judge,
    MiniCheckFactChecker,
    SummaryEvaluator,
    _coverage_score,
    _extract_json,
    build_evaluation_prompt,
)
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
        self.fact_check = {
            "sentence_labels": {"0": 1},
            "total_sentences": 1,
            "supported_sentences": 1,
            "unsupported_sentences": 0,
        }
        checker_patch = patch.object(MiniCheckFactChecker, "check", return_value=self.fact_check)
        self.checker = checker_patch.start()
        self.addCleanup(checker_patch.stop)

    def test_extract_json_ignores_trailing_model_output(self):
        self.assertEqual(
            json.loads(_extract_json('{"status": "ok"}\n{"extra": true}')),
            {"status": "ok"},
        )

    def test_judge_selection_prefers_local_model_and_falls_back_when_empty(self):
        with patch.dict(os.environ, {"OLLAMA_JUDGE_MODEL": "gpt-oss:20b", "GEMINI_API_KEY": ""}):
            evaluator = SummaryEvaluator({"model": "Summary"}, [self.packet])
            self.assertIsInstance(evaluator._judge, OllamaJudge)
            self.assertEqual(evaluator._judge.model, "gpt-oss:20b")
            self.assertEqual(evaluator._judge._provider.temperature, 0)
        for value in (None, "", "   "):
            with patch.dict(os.environ):
                os.environ.pop("OLLAMA_JUDGE_MODEL", None)
                if value is not None:
                    os.environ["OLLAMA_JUDGE_MODEL"] = value
                self.assertIsInstance(create_evaluation_judge(), GeminiJudge)
                self.assertNotIsInstance(create_evaluation_judge(), OllamaJudge)

    def test_local_judge_uses_existing_prompt_and_scoring(self):
        expected = FakeJudge().evaluate("Summary", self.packet)
        provider = Mock()
        provider.generate_text.return_value = json.dumps(expected)
        judge = OllamaJudge("gpt-oss:20b", provider=provider)
        result = SummaryEvaluator({"model": "Summary"}, [self.packet], judge=judge).evaluate()
        provider.generate_text.assert_called_once_with(
            build_evaluation_prompt("Summary", self.packet), "gpt-oss:20b",
            schema=JUDGE_RESPONSE_SCHEMA,
        )
        self.assertEqual(result["model"][0]["evaluation"]["avg_score"], 0.5196)

    def test_local_judge_errors_do_not_fall_back_to_gemini(self):
        provider = Mock()
        judge = OllamaJudge("gpt-oss:20b", provider=provider)
        provider.generate_text.side_effect = RuntimeError("offline")
        with self.assertRaisesRegex(EvaluationError, "Ollama judge request failed"):
            judge.evaluate("Summary", self.packet)
        provider.generate_text.side_effect = None
        provider.generate_text.return_value = "invalid JSON"
        with self.assertRaisesRegex(EvaluationError, "Ollama judge did not return valid JSON"):
            judge.evaluate("Summary", self.packet)

    def test_local_judge_does_not_silently_replace_inconsistent_denominator(self):
        invalid = FakeJudge().evaluate("Summary", self.packet)
        invalid["key_ideas"]["expected"] = 9
        provider = Mock()
        provider.generate_text.return_value = json.dumps(invalid)
        with self.assertRaisesRegex(EvaluationError, "expected=9"):
            OllamaJudge("gpt-oss:20b", provider=provider).evaluate("Summary", self.packet)
        self.assertEqual(provider.generate_text.call_count, 1)

    def test_local_cache_identity_tracks_provider_configuration(self):
        with patch.dict(os.environ, {"OLLAMA_JUDGE_MODEL": "gpt-oss:20b"}):
            judge = create_evaluation_judge()
            original = judge.cache_identity
            self.assertNotEqual(original, GeminiJudge(model=judge.model).cache_identity)
            judge.model = "another-model"
            self.assertNotEqual(original, judge.cache_identity)
            with patch.dict(os.environ, {"OLLAMA_CONTEXT_LENGTH": "16384"}):
                self.assertNotEqual(original, create_evaluation_judge().cache_identity)

    def test_judge_cache_identity_tracks_model_prompt_and_configuration(self):
        judge = GeminiJudge(model="judge-a", api_key="unused")
        original = judge.cache_identity

        judge.model = "judge-b"
        self.assertNotEqual(judge.cache_identity, original)

        judge.model = "judge-a"
        with patch(
            "llm.evaluator.build_evaluation_prompt",
            return_value="changed evaluation prompt",
        ):
            self.assertNotEqual(judge.cache_identity, original)

        with patch("llm.evaluator.JUDGE_TEMPERATURE", 0.25):
            self.assertNotEqual(judge.cache_identity, original)

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
            "fact_check": self.fact_check,
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
            "fact_checks": [
                {"packet_id": "complete", "fact_check": self.fact_check},
                {"packet_id": "empty", "fact_check": self.fact_check},
            ],
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

    def test_empty_incomplete_and_malformed_judgments_are_rejected(self):
        invalid_judgments = (
            {},
            {
                "key_ideas": {},
                "contributions": {},
                "hallucinated_claims": [],
                "total_summary_claims": 0,
            },
            {
                "key_ideas": {
                    "covered": 1,
                    "partial_covered": 0,
                    "expected": 1,
                    "missing": [],
                },
                "contributions": {
                    "covered": 0,
                    "partial_covered": 0,
                    "expected": 1,
                    "missing": [],
                },
                "hallucinated_claims": [],
                "total_summary_claims": 1,
            },
            {
                "key_ideas": {
                    "covered": "1",
                    "partial_covered": 0,
                    "expected": 1,
                    "missing": [],
                },
                "contributions": {
                    "covered": 1,
                    "partial_covered": 0,
                    "expected": 1,
                    "missing": [],
                },
                "hallucinated_claims": [],
                "total_summary_claims": 1,
            },
        )

        for judgment in invalid_judgments:
            with self.subTest(judgment=judgment):
                class InvalidJudge:
                    def evaluate(inner_self, summary, evidence_packet):
                        return judgment

                with self.assertRaises(EvaluationError):
                    SummaryEvaluator(
                        {"model-a": "Summary"},
                        [self.packet],
                        judge=InvalidJudge(),
                    ).evaluate()

    def test_zero_expected_dimensions_are_rejected_in_every_mode(self):
        valid_dimension = {
            "covered": 1,
            "partial_covered": 0,
            "expected": 1,
            "missing": [],
        }
        zero_dimension = {
            "covered": 0,
            "partial_covered": 0,
            "expected": 0,
            "missing": [],
        }

        for field in ("key_ideas", "contributions"):
            for mode in ("default", "detailed", "average"):
                with self.subTest(field=field, mode=mode):
                    judgment = {
                        "key_ideas": dict(valid_dimension),
                        "contributions": dict(valid_dimension),
                        "hallucinated_claims": [],
                        "total_summary_claims": 1,
                    }
                    judgment[field] = dict(zero_dimension)

                    class ZeroExpectedJudge:
                        def evaluate(inner_self, summary, evidence_packet):
                            return judgment

                    with self.assertRaisesRegex(
                        EvaluationError,
                        rf"{field}\.expected must be greater than zero",
                    ):
                        SummaryEvaluator(
                            {"model-a": "Summary"},
                            [self.packet],
                            judge=ZeroExpectedJudge(),
                        ).evaluate(mode)

    def test_coverage_score_defensively_rejects_zero_expected(self):
        with self.assertRaisesRegex(EvaluationError, "expected is not greater than zero"):
            _coverage_score({
                "covered": 0,
                "partial_covered": 0,
                "expected": 0,
            })

    def test_evaluation_prompt_requires_positive_expected_counts(self):
        prompt = build_evaluation_prompt("Summary", self.packet)

        self.assertIn("must never be zero", prompt)
        self.assertNotIn('"expected": 0', prompt)

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


class MiniCheckTests(unittest.TestCase):
    def test_missing_sentence_data_has_actionable_error(self):
        splitter = Mock(side_effect=LookupError("Missing punkt_tab"))
        checker = MiniCheckFactChecker(sentence_splitter=splitter)
        with self.assertRaisesRegex(EvaluationError, "nltk.downloader punkt punkt_tab"):
            checker.check("Claim.", EvidencePacket("p", "Evidence."))

    def test_scorer_failure_is_reported_as_evaluation_error(self):
        scorer = Mock()
        scorer.score.side_effect = RuntimeError("inference failed")
        checker = MiniCheckFactChecker(scorer=scorer, sentence_splitter=lambda _: ["Claim."])
        with self.assertRaisesRegex(EvaluationError, "MiniCheck fact checking failed"):
            checker.check("Claim.", EvidencePacket("p", "Evidence."))

    def test_checks_every_sentence_against_full_packet_and_serializes_labels(self):
        sentences = ["It works.", "It fails!", "It works.", "An unfinished tail"]
        scorer = Mock()
        scorer.score.return_value = ([1, 0, 1, 0], [], [], [])
        splitter = Mock(return_value=sentences)
        checker = MiniCheckFactChecker(scorer=scorer, sentence_splitter=splitter)
        packet = EvidencePacket("long", "Full packet. " * 1500)
        result = checker.check("Summary", packet)
        splitter.assert_called_once_with("Summary")
        scorer.score.assert_called_once_with(docs=[packet.text] * 4, claims=sentences)
        self.assertEqual(json.loads(json.dumps(result)), {
            "sentence_labels": {"0": 1, "1": 0, "2": 1, "3": 0},
            "total_sentences": 4,
            "supported_sentences": 2,
            "unsupported_sentences": 2,
        })

    def test_rejects_missing_and_invalid_labels(self):
        for labels in ([], [1, 0], [2], [0.5]):
            with self.subTest(labels=labels):
                scorer = Mock()
                scorer.score.return_value = (labels, [], [], [])
                checker = MiniCheckFactChecker(scorer=scorer, sentence_splitter=lambda _: ["Claim."])
                with self.assertRaises(EvaluationError):
                    checker.check("Claim.", EvidencePacket("p", "Evidence."))

    def test_initializes_requested_model_once(self):
        import types

        factory = Mock()
        module = types.ModuleType("minicheck.minicheck")
        module.MiniCheck = factory
        with patch.dict(sys.modules, {"minicheck": types.ModuleType("minicheck"), "minicheck.minicheck": module}):
            checker = MiniCheckFactChecker(cache_dir="/tmp/minicheck-test", batch_size=2)
            self.assertIs(checker._get_scorer(), checker._get_scorer())
        factory.assert_called_once_with(model_name="flan-t5-large", cache_dir="/tmp/minicheck-test", batch_size=2)

    def test_fact_check_runs_before_judge_without_changing_inputs_or_scores(self):
        calls = []
        packet = EvidencePacket("p", "Original packet.")
        class Checker:
            def check(self, summary, evidence_packet):
                calls.append(("check", summary, evidence_packet))
                return {"sentence_labels": {"0": 0}, "total_sentences": 1,
                        "supported_sentences": 0, "unsupported_sentences": 1}
        class Judge(FakeJudge):
            def evaluate(self, summary, evidence_packet):
                calls.append(("judge", summary, evidence_packet))
                return super().evaluate(summary, evidence_packet)
        for mode in ("default", "detailed", "average"):
            result = SummaryEvaluator({"m": "Original summary."}, [packet], judge=Judge(), fact_checker=Checker()).evaluate(mode)
            json.dumps(result, allow_nan=False)
            if mode == "default":
                self.assertEqual(result["m"][0]["evaluation"]["hallucination_score"], 0.1429)
                self.assertEqual(result["m"][0]["fact_check"]["unsupported_sentences"], 1)
        self.assertEqual(calls, [(stage, "Original summary.", packet)
                                 for _ in range(3) for stage in ("check", "judge")])


if __name__ == "__main__":
    unittest.main()
