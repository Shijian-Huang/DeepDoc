import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline
from llm import summarizer
from tools import benchmark


PAPER_TEXT = (
    "Abstract\n"
    "We introduce a sparse retrieval model for long scientific documents. "
    "The model improves recall on three benchmark datasets by 4.2 points. "
    "Training uses 100,000 annotated question pairs.\n\n"
    "Introduction\n"
    "Long documents remain difficult for dense retrievers because relevant passages are spread across many pages. "
    "We evaluate the approach on legal, medical and scientific collections."
)


def write_pdf(path: Path, text: str = PAPER_TEXT) -> str:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 760), text, fontsize=10)
    document.save(str(path))
    document.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_paper(pdf_url: str, sha256: str) -> dict:
    return {
        "id": "test:paper-1",
        "source": "test",
        "pdf_url": pdf_url,
        "sha256": sha256,
        "page_count": 1,
        "aclsum": {
            "challenge": "Dense retrievers struggle with long documents where relevant passages are spread across pages.",
            "approach": "They introduce a sparse retrieval model for long scientific documents.",
            "outcome": "The model improves recall on three benchmark datasets.",
            "source_sentences": {
                "challenge": ["Long documents remain difficult for dense retrievers because relevant passages are spread across many pages ."],
                "approach": ["We introduce a sparse retrieval model for long scientific documents ."],
                "outcome": ["The model improves recall on three benchmark datasets by 4.2 points ."],
            },
        },
    }


def make_result(**overrides) -> dict:
    summary = {
        "summary": (
            "Dense retrievers struggle with long documents because relevant passages are spread across pages. "
            "This paper presents a sparse retrieval model built for long scientific documents. "
            "It raises recall by 4.2 points across three benchmark datasets compared with earlier systems. "
            "The authors train it on 100,000 annotated question pairs."
        ),
        "summary_word_count": 40,
        "key_ideas": ["Sparse retrieval can handle long scientific documents."],
        "contributions": ["A retrieval model that improves recall on three benchmark datasets."],
        "evidence": [{"claim": "Recall improves by 4.2 points.", "source_ids": ["abstract_p1_01"], "pages": [1]}],
        "verified_facts": [{"fact": "Training uses 100,000 annotated question pairs."}],
        "diagnostics": {"filtered_sentences": [], "compact_retry": False, "deterministic_fallback": False, "verified_fact_count": 4},
    }
    summary.update(overrides)
    return {
        "document_summary": summary,
        "evidence_sources": [
            {"source_id": "abstract_p1_01", "section": "abstract", "pages": [1],
             "excerpt": "We introduce a sparse retrieval model for long scientific documents. "
                        "The model improves recall on three benchmark datasets by 4.2 points."},
        ],
        "pipeline_diagnostics": {"status": "completed"},
    }


class BenchmarkTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_pdf = self.root / "source.pdf"
        self.sha256 = write_pdf(self.source_pdf)
        self.paper = make_paper(self.source_pdf.as_uri(), self.sha256)
        self.paper_set = {"name": "test", "_sha256": "set-sha", "summary_mode": "standard", "papers": [self.paper]}
        self.paper_dir = self.root / "papers"
        self.run_dir = self.root / "runs"

    def tearDown(self):
        self.temp.cleanup()

    def paper_text(self):
        with redirect_stdout(StringIO()):
            benchmark.fetch_papers(self.paper_set, self.paper_dir, pause_seconds=0)
        return benchmark.PaperText(benchmark.paper_pdf_path(self.paper, self.paper_dir))

    def check(self, result, status="ok"):
        record = {"paper_id": self.paper["id"], "repeat": 1, "status": status, "result": result, "wall_seconds": 3.0}
        return benchmark.check_run(record, self.paper, self.paper_text())

    def categories(self, checked):
        return [issue["category"] for issue in checked["issues"]]


class FetchTests(BenchmarkTestCase):
    def test_downloads_and_verifies_checksum(self):
        with redirect_stdout(StringIO()):
            problems = benchmark.fetch_papers(self.paper_set, self.paper_dir, pause_seconds=0)
        self.assertEqual(problems, [])
        self.assertTrue(benchmark.paper_pdf_path(self.paper, self.paper_dir).exists())

    def test_reports_checksum_mismatch(self):
        self.paper["sha256"] = "0" * 64
        with redirect_stdout(StringIO()):
            problems = benchmark.fetch_papers(self.paper_set, self.paper_dir, pause_seconds=0)
        self.assertEqual(len(problems), 1)
        self.assertIn("does not match the pinned", problems[0])


class RunTests(BenchmarkTestCase):
    @contextmanager
    def fake_pipeline(self, side_effect):
        @contextmanager
        def selection(provider=None, model=None):
            yield provider, model
        with (
            patch.object(summarizer, "use_llm_selection", selection),
            patch.object(pipeline, "run_pipeline", side_effect=side_effect) as run_pipeline,
            redirect_stdout(StringIO()),
        ):
            yield run_pipeline

    def run_benchmark(self, **kwargs):
        return benchmark.run_benchmark(self.paper_set, "v1", "ollama", "qwen3:8b", repeats=2,
                                       paper_dir=self.paper_dir, run_dir=self.run_dir, **kwargs)

    def test_saves_every_repeat_with_manifest_and_resumes(self):
        with redirect_stdout(StringIO()):
            benchmark.fetch_papers(self.paper_set, self.paper_dir, pause_seconds=0)
        with self.fake_pipeline([make_result(), RuntimeError("model crashed")]):
            output_dir = self.run_benchmark()
        manifest = json.loads((output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["settings"]["model"], "qwen3:8b")
        self.assertEqual(manifest["settings"]["context_length"], summarizer.ollama_context_length)
        runs = sorted((output_dir / "test-paper-1").glob("run-*.json"))
        self.assertEqual([json.loads(path.read_text())["status"] for path in runs], ["ok", "error"])
        self.assertIn("model crashed", json.loads(runs[1].read_text())["error"])

        with self.fake_pipeline([make_result()]) as run_pipeline:
            self.run_benchmark()
        run_pipeline.assert_not_called()

    def test_refuses_to_mix_settings_under_one_tag(self):
        with redirect_stdout(StringIO()):
            benchmark.fetch_papers(self.paper_set, self.paper_dir, pause_seconds=0)
        with self.fake_pipeline([make_result(), make_result()]):
            self.run_benchmark()
        changed = copy.deepcopy(self.paper_set)
        changed["_sha256"] = "different"
        with self.fake_pipeline([]):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "different paper_set_sha256"):
                benchmark.run_benchmark(changed, "v1", "ollama", "qwen3:8b", repeats=2,
                                        paper_dir=self.paper_dir, run_dir=self.run_dir)

    def test_missing_pdf_is_reported_before_running(self):
        with self.fake_pipeline([]):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "Missing PDFs"):
                self.run_benchmark()


class CheckTests(BenchmarkTestCase):
    def test_faithful_output_has_no_number_or_evidence_issues(self):
        checked = self.check(make_result())
        self.assertEqual(checked["number_errors"], 0)
        self.assertEqual(checked["invalid_evidence"], 0)
        self.assertEqual(checked["empty_fields"], [])
        self.assertEqual(checked["aclsum_coverage"], 1.0)
        self.assertGreater(checked["abstract_coverage"], 0.5)

    def test_invented_number_is_flagged(self):
        checked = self.check(make_result(key_ideas=["The model reaches 97.5% precision."]))
        self.assertEqual(checked["number_errors"], 1)
        issue = next(issue for issue in checked["issues"] if issue["category"] == "number_not_in_paper")
        self.assertEqual(issue["field"], "key_ideas")
        self.assertIn("97.5", issue["reference"])

    def test_copied_sentence_and_padding_are_flagged(self):
        copied = "Training uses 100,000 annotated question pairs."
        checked = self.check(make_result(summary=copied + " It is a sparse retrieval model for long scientific documents."))
        self.assertIn("copied_from_paper", self.categories(checked))
        self.assertEqual(checked["summary_sentences_equal_to_fact"], 1)

    def test_empty_and_duplicated_fields(self):
        checked = self.check(make_result(
            contributions=[],
            key_ideas=["A sparse retrieval model for long documents.", "A sparse retrieval model for long scientific documents."],
        ))
        self.assertEqual(checked["empty_fields"], ["contributions"])
        self.assertEqual(checked["duplicate_pairs"], 1)

    def test_missing_aspect_is_attributed_to_input_selection(self):
        result = make_result(summary="The paper studies retrieval of long documents in many domains today.",
                             key_ideas=["Retrieval matters."], contributions=["A new benchmark split."])
        result["evidence_sources"][0]["excerpt"] = "Unrelated text about something else entirely."
        checked = self.check(result)
        outcome_issue = next(issue for issue in checked["issues"] if issue["field"] == "outcome")
        self.assertEqual(outcome_issue["category"], "key_content_not_in_model_input")
        self.assertEqual(outcome_issue["component"], "section selection")

    def test_missing_aspect_that_reached_the_input_points_to_extraction(self):
        result = make_result(summary="The paper studies retrieval of long documents in many domains today.",
                             key_ideas=["Retrieval matters."], contributions=["A new benchmark split."],
                             verified_facts=[])
        checked = self.check(result)
        outcome_issue = next(issue for issue in checked["issues"] if issue["field"] == "outcome")
        self.assertEqual(outcome_issue["category"], "key_content_not_extracted")

    def test_invalid_evidence_reference(self):
        checked = self.check(make_result(evidence=[{"claim": "Recall improves.", "source_ids": ["missing_source"], "pages": [1]}]))
        self.assertEqual(checked["invalid_evidence"], 1)
        checked = self.check(make_result(evidence=[{"claim": "Recall improves.", "source_ids": ["abstract_p1_01"], "pages": [4]}]))
        self.assertEqual(checked["invalid_evidence"], 1)

    def test_unsupported_evaluative_wording_and_field_counts(self):
        result = make_result(contributions=[], key_ideas=["A novel sparse retrieval model for long scientific documents."])
        result["document_summary"]["diagnostics"].update({"checked_sentence_count": 8, "filtered_sentences": [{}, {}]})
        checked = self.check(result)
        self.assertEqual(checked["unsupported_evaluative_words"], 1)
        self.assertIn("unsupported_evaluative_wording", self.categories(checked))
        self.assertTrue(checked["contributions_empty"])
        self.assertEqual((checked["key_idea_count"], checked["contribution_count"]), (1, 0))
        self.assertEqual(checked["filtered_share"], 0.25)

    def test_citation_metrics(self):
        result = make_result(citations={
            "summary": [{"text": "x", "pages": [1]}, {"text": "y", "pages": [9]}],
            "key_ideas": [None],
            "contributions": [],
        })
        result["document_summary"]["diagnostics"].update({"cited_sentence_count": 3, "uncited_sentence_count": 1})
        checked = self.check(result)
        self.assertEqual(checked["cited_share"], 0.75)
        self.assertEqual(checked["invalid_citation_pages"], 1)
        self.assertIsNone(self.check(make_result())["cited_share"])

    def test_failed_run_is_recorded(self):
        checked = self.check(None, status="error")
        self.assertTrue(checked["failed"])
        self.assertEqual(self.categories(checked), ["pipeline_failure"])


class CompareTests(unittest.TestCase):
    def checks(self, number_errors_by_repeat, run="v1/ollama-qwen3-8b"):
        runs = []
        for paper_index, values in enumerate(number_errors_by_repeat):
            for repeat, value in enumerate(values, start=1):
                runs.append({"paper_id": f"p{paper_index}", "repeat": repeat, "failed": False,
                             "number_errors": value, "issues": [], "empty_fields": []})
        manifest = {"git_sha": "abc1234567", "git_dirty": False, "repeats": len(number_errors_by_repeat[0]),
                    "settings": {"provider": "ollama", "model": "qwen3:8b"}, "paper_ids": []}
        return {"run": run, "manifest": manifest, "runs": runs}

    def row(self, report, label="Sentences with a number not in the paper"):
        return next(line for line in report.splitlines() if line.startswith(f"| {label}"))

    def test_consistent_improvement_beyond_noise(self):
        before = self.checks([[6, 7, 8], [5, 6, 7], [7, 7, 7]])
        after = self.checks([[1, 2, 2], [1, 1, 2], [2, 2, 2]], run="v2/ollama-qwen3-8b")
        self.assertIn("✅ better", self.row(benchmark.compare_checks(before, after)))

    def test_change_within_noise_is_not_claimed(self):
        before = self.checks([[2, 6, 4], [3, 7, 5]])
        after = self.checks([[3, 5, 3], [4, 6, 4]], run="v2/ollama-qwen3-8b")
        self.assertIn("≈ within noise", self.row(benchmark.compare_checks(before, after)))

    def test_regression_is_reported(self):
        before = self.checks([[1, 1], [1, 1], [0, 0]])
        after = self.checks([[4, 4], [5, 5], [3, 3]], run="v2/ollama-qwen3-8b")
        self.assertIn("⚠️ worse", self.row(benchmark.compare_checks(before, after)))

    def test_informational_metrics_have_no_verdict(self):
        before = self.checks([[1, 1]])
        after = self.checks([[2, 2]], run="v2/ollama-qwen3-8b")
        for checks, words in ((before, 150), (after, 200)):
            for run in checks["runs"]:
                run["summary_words"] = words
        self.assertIn("| 150.00 | 200.00 | - | info |", self.row(benchmark.compare_checks(before, after), "Summary words"))

    def test_report_lists_frequent_issues(self):
        checks = self.checks([[1]])
        checks["runs"][0]["issues"] = [{"paper_id": "p0", "category": "number_not_in_paper", "component": "generation"}] * 2
        report = benchmark.report_checks(checks)
        self.assertIn("| number_not_in_paper | generation | 2 | 1 |", report)


class PaperSetTests(unittest.TestCase):
    def test_committed_paper_sets_are_well_formed_and_disjoint(self):
        sets = {name: benchmark.load_paper_set(name) for name in ("dev", "holdout")}
        ids = {name: {paper["id"] for paper in paper_set["papers"]} for name, paper_set in sets.items()}
        self.assertFalse(ids["dev"] & ids["holdout"])
        for paper_set in sets.values():
            self.assertEqual(len(paper_set["papers"]), 10)
            for paper in paper_set["papers"]:
                self.assertRegex(paper["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(paper["pdf_url"].startswith("https://"))
                if paper["source"] == "acl_anthology":
                    self.assertEqual(set(paper["aclsum"]) - {"source_sentences"}, {"challenge", "approach", "outcome"})


if __name__ == "__main__":
    unittest.main()
