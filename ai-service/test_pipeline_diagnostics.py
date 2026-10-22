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

    def test_grounding_keeps_paraphrases_and_rejects_distortions(self):
        fact = {
            "fact_id": "fact_01",
            "fact": "The big transformer model outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU on WMT 2014 English-to-German, establishing a new state-of-the-art BLEU score of 28.4.",
            "source_quote": "On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.",
        }
        cases = {
            "The larger Transformer set a new record of 28.4 BLEU for English-German translation, beating earlier systems and ensembles by over 2.0 points.": True,
            "The big transformer model reaches 29.4 BLEU on the WMT 2014 English-to-German translation task.": False,
            "The big transformer model underperforms the best previously reported models on WMT 2014 English-to-German by 2.0 BLEU.": False,
            "This breakthrough fundamentally transforms how researchers approach machine translation tasks.": False,
        }
        for sentence, expected in cases.items():
            with self.subTest(sentence=sentence):
                self.assertEqual(bool(summarizer._supported_sentences(sentence, ["fact_01"], [fact])), expected)

    def test_high_risk_terms_match_whole_word_starts(self):
        source = "The model improves accuracy on three datasets compared with the baseline model."
        self.assertTrue(summarizer._claim_supported_by_text("The model improves accuracy on three datasets.", source))
        self.assertTrue(summarizer._claim_supported_by_text(
            "The model achieves improvements in accuracy on three datasets.", "The model achieves improvements in accuracy on three datasets."
        ))
        self.assertFalse(summarizer._claim_supported_by_text("The model proves accuracy gains on three datasets.", source))
        self.assertFalse(summarizer._claim_supported_by_text("The novelty is better accuracy on three datasets.", source))

    def test_summary_prompt_discourages_unsupported_evaluative_words(self):
        prompt = summarizer.build_research_summary_prompt([], "standard")
        self.assertIn("Do not use evaluative words such as novel", prompt)
        self.assertNotIn("genuinely new", prompt)

    def test_citations_map_final_sentences_to_sources(self):
        problem = "Long documents are hard for dense retrievers because relevant passages are spread across pages."
        method = "The paper proposes a sparse retriever that indexes each section of a long document separately."
        result_text = "The sparse retriever improves recall by 4.2 points on three benchmark datasets."
        sources = [
            {"source_id": "intro_p1_01", "section": "introduction", "pages": [1, 2], "excerpt": f"{problem} {method}"},
            {"source_id": "results_p5_01", "section": "results", "pages": [5], "excerpt": result_text},
        ]
        facts = {"facts": [
            {"category": "research_question", "fact": problem, "source_ids": ["intro_p1_01"], "source_quote": problem},
            {"category": "method", "fact": method, "source_ids": ["intro_p1_01"], "source_quote": method},
            {"category": "result", "fact": result_text, "source_ids": ["results_p5_01"], "source_quote": result_text},
        ]}
        analysis = {
            "summary_paragraphs": [
                {"text": "Dense retrievers struggle with long documents because relevant passages are spread across pages.", "fact_ids": ["fact_01"]},
                {"text": "This breakthrough transforms how everyone reads papers.", "fact_ids": ["fact_02"]},
            ],
            "summary_word_count": 0,
            "key_ideas": [{"text": "A sparse retriever indexes each section of a long document separately.", "fact_ids": ["fact_02"]}],
            "contributions": [],
            "evidence": [],
        }
        with patch.object(summarizer, "generate_json", side_effect=[facts, analysis]):
            result = summarizer.summarize_research_paper("packet", summary_mode="paragraph", evidence_sources=sources)

        citations = result["citations"]
        self.assertNotIn("breakthrough", result["summary"])
        first = citations["summary"][0]
        self.assertEqual(first["text"], "Dense retrievers struggle with long documents because relevant passages are spread across pages.")
        self.assertEqual((first["origin"], first["fact_ids"], first["source_ids"], first["pages"]), ("generated", ["fact_01"], ["intro_p1_01"], [1, 2]))
        padded = [citation for citation in citations["summary"] if citation["origin"] == "source_fact"]
        self.assertTrue(padded)
        self.assertTrue(all(citation["text"] in result["summary"] for citation in citations["summary"]))
        self.assertEqual(citations["key_ideas"][0]["source_ids"], ["intro_p1_01"])
        self.assertEqual(citations["contributions"], [])
        self.assertEqual(result["diagnostics"]["uncited_sentence_count"], 0)
        self.assertEqual(result["diagnostics"]["cited_sentence_count"], len(citations["summary"]) + 1)

    def test_citation_pages_narrow_to_the_page_containing_the_quote(self):
        citations = {"summary": [{"quotes": ["The sparse retriever improves recall by 4.2 points."], "pages": [3, 4, 5]}],
                     "key_ideas": [None], "contributions": []}
        pages = [
            {"page": 3, "text": "Method details."},
            {"page": 4, "text": "Results: The sparse retriever improves\nrecall by 4.2 points."},
            {"page": 5, "text": "Conclusion."},
        ]
        pipeline._locate_citation_pages(citations, pages)
        self.assertEqual(citations["summary"][0]["pages"], [4])
        self.assertEqual(citations["summary"][0]["quote_pages"], [[4]])
        self.assertEqual(citations["summary"][0]["page_source"], "quote")

    def test_citation_quotes_on_different_pages_keep_their_own_pages(self):
        citation = {"quotes": ["Models (2) and (3) show no significant difference in time.", "One participant was surprised by how little code they read."],
                    "pages": [3, 4, 5, 6, 7, 8, 9, 10]}
        pages = [
            {"page": 6, "text": "Models (2) and (3) show no significant difference in time."},
            {"page": 10, "text": "One participant was surprised by how little code they read."},
        ]
        pipeline._locate_citation_pages({"summary": [citation]}, pages)
        self.assertEqual(citation["quote_pages"], [[6], [10]])
        self.assertEqual(citation["pages"], [6, 10])

    def test_retry_and_fallback_are_recorded(self):
        text = "The study evaluates a retrieval method on three datasets."
        sources = [{"source_id": "p1", "excerpt": text, "pages": [1], "section": "method"}]
        facts = {"facts": [{"fact": text, "source_quote": text, "source_ids": ["p1"], "category": "method"}]}
        invalid = json.JSONDecodeError("invalid", "{", 0)
        with patch.object(summarizer, "generate_json", side_effect=[invalid, facts, invalid]):
            result = summarizer.summarize_research_paper(text, evidence_sources=sources)
        diagnostics = result["diagnostics"]
        self.assertTrue(diagnostics["compact_retry"])
        self.assertEqual(diagnostics["compact_retry_reason"], "invalid_json")
        self.assertTrue(diagnostics["deterministic_fallback"])
        self.assertEqual(diagnostics["accepted_generated_fact_count"], 1)
        self.assertEqual(diagnostics["checked_sentence_count"], 0)
        self.assertIn(text, result["summary"])

    def test_malformed_fact_payloads_fall_back_to_source_facts(self):
        text = "The study evaluates a retrieval method on three datasets and reports accuracy gains."
        sources = [{"source_id": "p1", "excerpt": text, "pages": [1], "section": "method"}]
        for payload in ([], None, "facts", {"facts": None}, {"facts": "not a list"}, {"facts": {"fact": text}}):
            with self.subTest(payload=payload):
                diagnostics = {}
                verified = summarizer.normalize_verified_facts(payload, sources, diagnostics=diagnostics)
                self.assertEqual(diagnostics["generated_fact_count"], 0)
                self.assertEqual(diagnostics["accepted_generated_fact_count"], 0)
                self.assertEqual(diagnostics["verified_fact_count"], len(verified))
                self.assertEqual(diagnostics["source_supplement_count"], len(verified))
                self.assertEqual([fact["source_quote"] for fact in verified], [text])

    def test_malformed_fact_payload_does_not_fail_analysis(self):
        text = "The study evaluates a retrieval method on three datasets and reports accuracy gains."
        sources = [{"source_id": "p1", "excerpt": text, "pages": [1], "section": "method"}]
        invalid = json.JSONDecodeError("invalid", "{", 0)
        with patch.object(summarizer, "generate_json", side_effect=[[], invalid]):
            result = summarizer.summarize_research_paper(text, evidence_sources=sources)
        self.assertNotIn("error", result)
        self.assertEqual(result["diagnostics"]["generated_fact_count"], 0)
        self.assertFalse(result["diagnostics"]["compact_retry"])
        self.assertIsNone(result["diagnostics"]["compact_retry_reason"])

    def test_pipeline_records_stage_timings(self):
        with (
            patch.object(pipeline, "parse_document_pages", return_value=[{"text": "paper text"}]),
            patch.object(pipeline, "extract_document_title", return_value="Title"),
            patch.object(pipeline, "build_summary_input_from_pages", return_value=("text " * 50, ["method"], [{}])),
            patch.object(pipeline, "summarize_research_paper", return_value={"summary": "Summary"}),
            patch.object(pipeline, "_reference_result_fields", return_value={"references": []}),
            patch.object(pipeline, "active_llm_provider", return_value="ollama"),
            patch.object(pipeline, "active_llm_model", return_value="qwen3:8b"),
        ):
            result = pipeline.run_pipeline("paper.pdf")
        diagnostics = result["pipeline_diagnostics"]
        self.assertEqual(diagnostics["status"], "completed")
        self.assertEqual(diagnostics["llm_provider"], "ollama")
        self.assertEqual(diagnostics["llm_model"], "qwen3:8b")
        self.assertEqual(set(diagnostics["stage_seconds"]), {"parse", "section_selection", "summary", "references"})
        self.assertTrue(all(value >= 0 for value in diagnostics["stage_seconds"].values()))

    def test_pipeline_early_return_records_llm_selection(self):
        with (
            patch.object(pipeline, "parse_document_pages", return_value=[{"text": ""}]),
            patch.object(pipeline, "extract_document_title", return_value="Title"),
            patch.object(pipeline, "active_llm_provider", return_value="gemini"),
            patch.object(pipeline, "active_llm_model", return_value="gemini-2.5-flash-lite"),
        ):
            result = pipeline.run_pipeline("paper.pdf")
        diagnostics = result["pipeline_diagnostics"]
        self.assertEqual(diagnostics["status"], "no_readable_text")
        self.assertEqual(diagnostics["llm_provider"], "gemini")
        self.assertEqual(diagnostics["llm_model"], "gemini-2.5-flash-lite")


class CompactRetryReasonTests(unittest.TestCase):
    TEXT = "The study evaluates a retrieval method on three datasets and reports accuracy gains."

    def run_with_provider_responses(self, responses):
        sources = [{"source_id": "p1", "excerpt": self.TEXT, "pages": [1], "section": "method"}]
        facts = {"facts": [{"fact": self.TEXT, "source_quote": self.TEXT, "source_ids": ["p1"], "category": "method"}]}
        responses = [json.dumps(facts) if item == "FACTS" else item for item in responses]
        with (
            patch.object(summarizer, "require_llm_configuration"),
            patch.object(summarizer, "wait_for_rate_limit"),
            patch.object(summarizer, "active_llm_provider", return_value="gemini"),
            patch.object(summarizer, "_gemini_provider") as provider,
        ):
            provider.return_value.generate_text.side_effect = responses
            result = summarizer.summarize_research_paper(self.TEXT, evidence_sources=sources)
            call_count = provider.return_value.generate_text.call_count
        return result["diagnostics"], call_count

    def test_provider_failure_is_reported_and_still_retried(self):
        diagnostics, calls = self.run_with_provider_responses([RuntimeError("HTTP 503"), "FACTS", "not json"])
        self.assertTrue(diagnostics["compact_retry"])
        self.assertEqual(diagnostics["compact_retry_reason"], "provider_error")
        self.assertEqual(calls, 3)

    def test_malformed_json_is_reported_as_invalid_json(self):
        diagnostics, calls = self.run_with_provider_responses(["not json", "FACTS", "not json"])
        self.assertTrue(diagnostics["compact_retry"])
        self.assertEqual(diagnostics["compact_retry_reason"], "invalid_json")
        self.assertEqual(calls, 3)

    def test_no_retry_reports_no_reason(self):
        diagnostics, calls = self.run_with_provider_responses(["FACTS", "not json"])
        self.assertFalse(diagnostics["compact_retry"])
        self.assertIsNone(diagnostics["compact_retry_reason"])
        self.assertEqual(calls, 2)

    def test_provider_error_remains_a_json_decode_error_with_detail(self):
        with (
            patch.object(summarizer, "require_llm_configuration"),
            patch.object(summarizer, "wait_for_rate_limit"),
            patch.object(summarizer, "active_llm_provider", return_value="gemini"),
            patch.object(summarizer, "_gemini_provider") as provider,
        ):
            provider.return_value.generate_text.side_effect = RuntimeError("HTTP 503")
            with self.assertRaises(json.JSONDecodeError) as caught:
                summarizer.generate_json("prompt")
        self.assertIsInstance(caught.exception, summarizer.LLMProviderError)
        self.assertEqual(caught.exception.doc, "HTTP 503")


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
        self.assertNotIsInstance(caught.exception, summarizer.LLMProviderError)

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
