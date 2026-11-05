import copy
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from llm.evaluator import EvaluationError, build_evaluation_prompt, evaluate_pipeline_output


class PipelineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((Path(__file__).parent / 'data/benchmark_runs/v1/ollama-qwen3-8b/acl-2020.acl-main.642/run-1.json').read_text())
        self.judge = Mock()

        def assess(candidate, packet):
            dimension = {'covered': int(bool(candidate)), 'partial_covered': 0,
                         'expected': 1, 'missing': [] if candidate else ['Missing central content']}
            return {'key_ideas': copy.deepcopy(dimension), 'contributions': copy.deepcopy(dimension),
                    'hallucinated_claims': [], 'total_summary_claims': int(bool(candidate))}
        self.judge.evaluate.side_effect = assess

    def test_reference_input_evaluates_each_field_against_excerpts(self):
        result = evaluate_pipeline_output(self.payload, judge=self.judge)['fields']
        self.assertEqual(set(result), {'key_ideas', 'contributions', 'evidence', 'summary'})
        self.assertEqual(result['contributions']['completeness'], 0)
        self.assertIsNone(result['contributions']['faithfulness'])
        self.assertEqual(result['summary']['faithfulness'], 1)
        calls = self.judge.evaluate.call_args_list
        self.assertEqual(len(calls), 4)
        for call in calls:
            candidate, packet = call.args
            self.assertEqual(json.loads(packet.text)[0]['excerpt'], self.payload['result']['evidence_sources'][0]['excerpt'])
            self.assertIn('document_summary.' + packet.evaluation_field, build_evaluation_prompt(candidate, packet))
        self.assertEqual(calls[2].args[0].splitlines()[0], self.payload['result']['document_summary']['evidence'][0]['claim'])
        self.assertEqual(calls[3].args[0], self.payload['result']['document_summary']['summary'])

    def test_supported_wrappers(self):
        for payload in (self.payload['result'], {'results': self.payload['result']}):
            self.assertEqual(len(evaluate_pipeline_output(payload, judge=self.judge)['fields']), 4)

    def test_invalid_input_fails_before_judge(self):
        self.payload['result']['evidence_sources'][0]['excerpt'] = ''
        with self.assertRaises(EvaluationError):
            evaluate_pipeline_output(self.payload, judge=self.judge)
        self.judge.evaluate.assert_not_called()

    def test_unsupported_claim_reduces_faithfulness(self):
        original = self.judge.evaluate.side_effect
        def assess(candidate, packet):
            result = original(candidate, packet)
            if candidate:
                result['hallucinated_claims'] = [{'claim': 'Unsupported', 'reason': 'Absent', 'evidence': None}]
            return result
        self.judge.evaluate.side_effect = assess
        result = evaluate_pipeline_output(self.payload, judge=self.judge)
        self.assertEqual(result['fields']['summary']['faithfulness'], 0)
