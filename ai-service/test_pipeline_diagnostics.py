import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import summarizer
from llm.cancellation import AnalysisCancellationToken, AnalysisCancelled, use_analysis_cancellation
from llm.providers.gemini import GeminiProvider
from llm.providers.ollama import OllamaProvider
from google.genai import errors
import pipeline


class PipelineDiagnosticsTests(unittest.TestCase):
    def test_summary_dedupe_retains_distinct_short_terms(self):
        self.assertFalse(summarizer._is_near_duplicate("GPU inference", "CPU inference"))
        self.assertTrue(summarizer._is_near_duplicate("The method improves retrieval", "This approach improves retrieval"))

    def test_filtered_sentence_records_field_and_missing_citation(self):
        diagnostics = {}
        result = summarizer.normalize_grounded_analysis_result({
            "summary_paragraphs": [{"text": "A claim without a citation.", "fact_ids": ["unknown"]}],
            "contributions": [{"text": "A novel contribution.", "fact_ids": []}],
        }, [], diagnostics=diagnostics)
        self.assertEqual(result["summary"], "")
        self.assertEqual(diagnostics["checked_sentence_count"], 2)
        self.assertEqual([item["field"] for item in diagnostics["filtered_sentences"]], ["summary", "contributions"])
        self.assertEqual(diagnostics["filtered_sentences"][0]["reason"], "no_valid_cited_facts")

    def test_grounding_number_and_comparison_boundaries(self):
        source = "The model achieved 92% accuracy on the dataset."
        self.assertTrue(summarizer._claim_supported_by_text("The model achieved 92 percent accuracy.", source))
        self.assertTrue(summarizer._claim_supported_by_text("The model achieved 92 % accuracy.", source))
        self.assertFalse(summarizer._claim_supported_by_text("The model achieved 92 accuracy.", source))
        self.assertFalse(summarizer._claim_supported_by_text("The model achieved 95% accuracy.", source))
        self.assertFalse(summarizer._claim_supported_by_text("The model achieved the highest accuracy.", source))

    def test_retry_and_fallback_are_recorded(self):
        text = "The study evaluates a retrieval method on three datasets."
        sources = [{"source_id": "p1", "excerpt": text, "pages": [1], "section": "method"}]
        facts = {"facts": [{"fact": text, "source_quote": text, "source_ids": ["p1"], "category": "method"}]}
        invalid = json.JSONDecodeError("invalid", "{", 0)
        with patch.object(summarizer, "generate_json", side_effect=[invalid, facts, invalid]):
            result = summarizer.summarize_research_paper(text, evidence_sources=sources)
        diagnostics = result["diagnostics"]
        self.assertTrue(diagnostics["compact_retry"])
        self.assertTrue(diagnostics["deterministic_fallback"])
        self.assertEqual(diagnostics["accepted_generated_fact_count"], 1)
        self.assertIn(text, result["summary"])

    def test_pipeline_records_stage_timings(self):
        with (
            patch.object(pipeline, "parse_document_pages", return_value=[{"text": "paper text"}]),
            patch.object(pipeline, "extract_document_title", return_value="Title"),
            patch.object(pipeline, "build_summary_input_from_pages", return_value=("text " * 50, ["method"], [{}])),
            patch.object(pipeline, "summarize_research_paper", return_value={"summary": "Summary"}),
            patch.object(pipeline, "_reference_result_fields", return_value={"references": []}),
        ):
            result = pipeline.run_pipeline("paper.pdf")
        diagnostics = result["pipeline_diagnostics"]
        self.assertEqual(diagnostics["status"], "completed")
        self.assertEqual(set(diagnostics["stage_seconds"]), {"parse", "section_selection", "summary", "references"})
        self.assertTrue(all(value >= 0 for value in diagnostics["stage_seconds"].values()))


class ProviderFailureTests(unittest.TestCase):
    def ollama(self):
        return OllamaProvider("http://localhost:11434", 12288, 3072, "10m", False, False)

    def test_gemini_sdk_error_is_reported(self):
        provider = GeminiProvider(None)
        provider.client = Mock()
        provider.client.models.generate_content.side_effect = errors.ClientError(429, {"error": {"message": "rate limited"}})
        with self.assertRaisesRegex(RuntimeError, "rate limited"):
            provider.generate_text("prompt", "test")

    def test_invalid_gemini_json_retains_raw_response(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text="invalid json")
        with (
            patch.object(summarizer, "client", client),
            patch.object(summarizer, "api_key", "test"),
            patch.object(summarizer, "llm_provider", "gemini"),
            patch.object(summarizer, "wait_for_rate_limit"),
        ):
            with self.assertRaises(json.JSONDecodeError) as caught:
                summarizer.generate_json("prompt")
        self.assertEqual(caught.exception.doc, "invalid json")

    def test_ollama_stream_failure_is_not_a_partial_success(self):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value = response
        def stream():
            yield b'{"message":{"content":"partial"}}\n'
            raise OSError("connection interrupted")
        response.__iter__.side_effect = stream
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "connection interrupted"):
                self.ollama().generate_text("prompt", "qwen")

    def test_cancelled_stream_closes_response_and_restores_context(self):
        from unittest.mock import MagicMock
        token = AnalysisCancellationToken()
        response = MagicMock()
        response.__enter__.return_value = response
        def stream():
            token.cancel()
            yield b'{"message":{"content":"partial"}}\n'
        response.__iter__.side_effect = stream
        with patch("urllib.request.urlopen", return_value=response), use_analysis_cancellation(token):
            with self.assertRaises(AnalysisCancelled):
                self.ollama().generate_text("prompt", "qwen")
        response.close.assert_called_once()
        summarizer.check_analysis_cancelled()
        self.assertIsNone(token._response)
