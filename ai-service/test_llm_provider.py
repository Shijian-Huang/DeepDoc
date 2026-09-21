import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import summarizer


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LlmProviderTests(unittest.TestCase):
    def test_ollama_configuration(self):
        with (
            patch.object(summarizer, "llm_provider", "ollama"),
            patch.object(summarizer, "ollama_base_url", "http://127.0.0.1:11434"),
            patch.object(summarizer, "ollama_models", ["qwen3:4b"]),
            patch.object(summarizer, "ollama_cpu_only", False),
        ):
            self.assertTrue(summarizer.is_llm_configured())
            self.assertEqual(summarizer.active_llm_model(), "qwen3:4b")

    def test_generate_json_uses_ollama_chat_endpoint(self):
        response = _FakeResponse({"message": {"content": '{"summary": "ok"}'}})
        with (
            patch.object(summarizer, "llm_provider", "ollama"),
            patch.object(summarizer, "ollama_models", ["qwen3:4b"]),
            patch.object(summarizer, "ollama_cpu_only", False),
            patch.object(summarizer, "wait_for_rate_limit"),
            patch("urllib.request.urlopen", return_value=response) as urlopen,
        ):
            result = summarizer.generate_json("Return JSON")

        self.assertEqual(result, {"summary": "ok"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:4b")
        self.assertEqual(body["format"], "json")
        self.assertTrue(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["keep_alive"], "10m")
        self.assertEqual(body["options"]["temperature"], 0.1)
        self.assertEqual(body["options"]["num_ctx"], 12288)
        self.assertEqual(body["options"]["num_predict"], 3072)
        self.assertNotIn("num_gpu", body["options"])

    def test_generate_json_can_force_cpu_only(self):
        response = _FakeResponse({"message": {"content": '{"summary": "ok"}'}})
        with (
            patch.object(summarizer, "llm_provider", "ollama"),
            patch.object(summarizer, "ollama_models", ["qwen3:4b"]),
            patch.object(summarizer, "ollama_cpu_only", True),
            patch.object(summarizer, "wait_for_rate_limit"),
            patch("urllib.request.urlopen", return_value=response) as urlopen,
        ):
            summarizer.generate_json("Return JSON")

        body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["options"]["num_gpu"], 0)

    def test_generate_json_passes_schema_to_ollama(self):
        response = _FakeResponse({"message": {"content": '{"facts": []}'}})
        schema = {"type": "object", "properties": {"facts": {"type": "array"}}}
        with (
            patch.object(summarizer, "llm_provider", "ollama"),
            patch.object(summarizer, "ollama_models", ["qwen3:4b"]),
            patch.object(summarizer, "wait_for_rate_limit"),
            patch("urllib.request.urlopen", return_value=response) as urlopen,
        ):
            summarizer.generate_json("Return facts", schema=schema)

        body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["format"], schema)

    def test_fact_verification_rejects_unknown_sources_and_changed_numbers(self):
        sources = [{
            "source_id": "results_p4_01",
            "section": "results",
            "pages": [4],
            "excerpt": "Five models passed reliability filtering. The average cosine similarity was 0.24.",
        }]
        payload = {"facts": [
            {
                "category": "result",
                "fact": "Five models passed reliability filtering.",
                "source_ids": ["results_p4_01"],
                "source_quote": "Five models passed reliability filtering.",
            },
            {
                "category": "result",
                "fact": "Seven models passed reliability filtering.",
                "source_ids": ["results_p4_01"],
                "source_quote": "Five models passed reliability filtering.",
            },
            {
                "category": "result",
                "fact": "The average cosine similarity was 0.24.",
                "source_ids": ["missing_source"],
                "source_quote": "The average cosine similarity was 0.24.",
            },
        ]}

        verified = summarizer.normalize_verified_facts(payload, sources)

        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["fact"], "Five models passed reliability filtering.")

    def test_grounded_evidence_uses_pages_from_real_source(self):
        sources = [{
            "source_id": "results_p4_01",
            "section": "results",
            "pages": [4],
            "excerpt": "The average cosine similarity was 0.24.",
        }]
        facts = [{
            "category": "result",
            "fact": "The average cosine similarity was 0.24.",
            "source_ids": ["results_p4_01"],
            "source_quote": "The average cosine similarity was 0.24.",
        }]
        result = {"evidence": [{
            "claim": "The average cosine similarity was 0.24.",
            "source_ids": ["results_p4_01"],
            "section": "abstract",
            "pages": [1],
        }]}

        summarizer._normalize_grounded_evidence(result, facts, sources)

        self.assertEqual(result["evidence"][0]["section"], "results")
        self.assertEqual(result["evidence"][0]["pages"], [4])

    def test_grounded_analysis_removes_unsupported_sentences(self):
        facts = [{
            "fact_id": "fact_01",
            "category": "result",
            "fact": "Seven LLMs produced valid responses.",
            "source_ids": ["experiment_p2_01"],
            "source_quote": "Seven LLMs produced valid responses.",
        }]
        result = {
            "summary_paragraphs": [{
                "text": "Seven LLMs produced valid responses. This novel framework enables responsible deployment.",
                "fact_ids": ["fact_01"],
            }],
            "key_ideas": [{"text": "Seven LLMs produced valid responses.", "fact_ids": ["fact_01"]}],
            "contributions": [{"text": "The paper introduces a novel framework.", "fact_ids": ["fact_01"]}],
        }

        grounded = summarizer.normalize_grounded_analysis_result(result, facts)

        self.assertEqual(grounded["summary"], "Seven LLMs produced valid responses.")
        self.assertEqual(grounded["key_ideas"], ["Seven LLMs produced valid responses."])
        self.assertEqual(grounded["contributions"], [])

    def test_short_summary_is_extended_only_with_verified_facts(self):
        result = {"summary": "The study evaluates model consistency."}
        facts = [{
            "fact_id": "fact_01",
            "category": "result",
            "fact": "Five models passed reliability filtering with an average score of 0.24. " * 4,
            "source_ids": ["results_p4_01"],
            "source_quote": "Five models passed reliability filtering with an average score of 0.24.",
        }]

        summarizer.supplement_short_summary(result, facts, "standard")

        self.assertIn("Five models passed reliability filtering", result["summary"])
        self.assertTrue(result["summary"].endswith("."))

    def test_unsupported_superlative_is_rejected(self):
        self.assertFalse(summarizer._claim_supported_by_text(
            "ChatGLM3 had the lowest cosine similarity at 0.24.",
            "ChatGLM3 had a cosine similarity of 0.24.",
            min_overlap=0.5,
        ))

    def test_short_summary_is_not_split_in_the_middle_of_a_sentence(self):
        summary = "The method extracts verified facts and then writes a summary using those facts."

        normalized = summarizer._enforce_summary_paragraphs(summary, "standard")

        self.assertEqual(normalized, summary)

    def test_generated_sentence_cleanup_preserves_grammar(self):
        self.assertEqual(
            summarizer._clean_generated_sentence("The method writes a: summary using verified facts"),
            "The method writes a summary using verified facts.",
        )

    def test_general_source_fact_fallback_uses_exact_sentences(self):
        sources = [{
            "source_id": "abstract_p1_01",
            "section": "abstract",
            "pages": [1],
            "excerpt": (
                "This paper studies a retrieval method for long scientific documents. "
                "The method is evaluated on three public datasets and improves recall. "
                "The authors also discuss limitations of the current evaluation protocol."
            ),
        }]

        facts = summarizer.normalize_verified_facts({"facts": []}, sources)

        self.assertGreaterEqual(len(facts), 3)
        self.assertEqual(facts[0]["fact"], facts[0]["source_quote"])
        self.assertEqual(facts[0]["source_ids"], ["abstract_p1_01"])

    def test_analysis_json_failure_falls_back_to_verified_facts(self):
        sources = [{
            "source_id": "abstract_p1_01",
            "section": "abstract",
            "pages": [1],
            "excerpt": "The study evaluates a document retrieval method on three public datasets.",
        }]
        fact_response = {"facts": [{
            "category": "method",
            "fact": "The study evaluates a document retrieval method on three public datasets.",
            "source_ids": ["abstract_p1_01"],
            "source_quote": "The study evaluates a document retrieval method on three public datasets.",
        }]}
        invalid_json = json.JSONDecodeError("invalid", "", 0)
        with patch.object(summarizer, "generate_json", side_effect=[fact_response, invalid_json]):
            result = summarizer.summarize_research_paper(
                "[SOURCE_ID: abstract_p1_01]",
                summary_mode="paragraph",
                evidence_sources=sources,
            )

        self.assertIn("document retrieval method", result["summary"])
        self.assertTrue(result["faithfulness"]["deterministic_fallback"])

    def test_fact_extraction_retries_with_compact_prompt(self):
        sources = [{
            "source_id": "abstract_p1_01",
            "section": "abstract",
            "pages": [1],
            "excerpt": "The study evaluates a retrieval method on three datasets.",
        }]
        valid = {"facts": [{
            "category": "method",
            "fact": "The study evaluates a retrieval method on three datasets.",
            "source_ids": ["abstract_p1_01"],
            "source_quote": "The study evaluates a retrieval method on three datasets.",
        }]}
        invalid = json.JSONDecodeError("incomplete", "{", 1)
        with patch.object(summarizer, "generate_json", side_effect=[invalid, valid]) as generate:
            facts = summarizer.extract_verified_facts("source packet", sources)

        self.assertEqual(len(facts), 1)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("compact retry", generate.call_args.args[0])

    def test_unknown_provider_is_rejected(self):
        with patch.object(summarizer, "llm_provider", "unknown"):
            self.assertFalse(summarizer.is_llm_configured())
            self.assertIn("Unsupported LLM_PROVIDER", summarizer.llm_configuration_error())

    def test_ollama_connection_checks_installed_model(self):
        response = _FakeResponse({"models": [{"name": "qwen3:8b"}]})
        with (
            patch.object(summarizer, "llm_provider", "ollama"),
            patch.object(summarizer, "ollama_models", ["qwen3:8b"]),
            patch("urllib.request.urlopen", return_value=response),
        ):
            self.assertTrue(summarizer.is_llm_connected())

    def test_request_selection_is_isolated_and_restored(self):
        options = [{
            "provider": "gemini",
            "model": "gemini-test",
            "label": "Gemini",
            "available": True,
            "selected": True,
        }]
        original_provider = summarizer.active_llm_provider()
        original_model = summarizer.active_llm_model()
        with patch.object(summarizer, "llm_options", return_value=options):
            with summarizer.use_llm_selection("gemini", "gemini-test"):
                self.assertEqual(summarizer.active_llm_provider(), "gemini")
                self.assertEqual(summarizer.active_llm_model(), "gemini-test")
        self.assertEqual(summarizer.active_llm_provider(), original_provider)
        self.assertEqual(summarizer.active_llm_model(), original_model)


if __name__ == "__main__":
    unittest.main()
