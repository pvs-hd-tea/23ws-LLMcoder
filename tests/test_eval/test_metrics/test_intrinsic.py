# Generated with GPT-4 under supervision

import unittest
from unittest.mock import patch

from llmcoder.eval.metrics.intrinsic import agility_score, all_analyzers_passed_score, loops_required_score, time_score, tokens_used_score


class TestLoopsRequiredScore(unittest.TestCase):
    def setUp(self) -> None:
        # Typical llmcoder result
        self.typical_result = {
            'messages': [
                {'role': 'system', 'content': 'system prompt'},
                {'role': 'user', 'content': 'input code'},
                {'role': 'system', 'content': 'first completion'},
                {'role': 'user', 'content': 'feedback prompt'},
                {'role': 'system', 'content': 'second completion'},
                {'role': 'user', 'content': 'feedback prompt 2'},
                {'role': 'system', 'content': 'third completion, passing'}
            ]
        }

        # Empty conversation
        self.empty_result: dict[str, list] = {
            'messages': []
        }

        # Odd number of messages
        self.odd_messages_result = {
            'messages': [
                {'role': 'system', 'content': 'system prompt'},
                {'role': 'user', 'content': 'input code'},
                {'role': 'system', 'content': 'first completion'},
                {'role': 'user', 'content': 'feedback prompt'},
            ]
        }

    def test_loops_required_score_typical(self) -> None:
        expected_loops = 2  # (7 messages - 3) // 2
        self.assertEqual(loops_required_score(self.typical_result), expected_loops)

    def test_loops_required_score_empty(self) -> None:
        expected_loops = 0
        self.assertEqual(loops_required_score(self.empty_result), expected_loops)

    def test_loops_required_score_odd_messages(self) -> None:
        expected_loops = 0  # (4 messages - 3) // 2
        self.assertEqual(loops_required_score(self.odd_messages_result), expected_loops)


class TestTokensUsedScore(unittest.TestCase):
    def setUp(self) -> None:
        # Mocking the tokenizer
        self.mock_tokenizer = patch('tiktoken.get_encoding')
        self.mock_tokenizer.start().return_value.encode.side_effect = lambda text: text.split()

        # Typical llmcoder result
        self.typical_result = {
            'messages': [
                {'role': 'user', 'content': 'Hello world!'},
                {'role': 'assistant', 'content': 'How are you?'},
                {'role': 'user', 'content': 'I am fine.'}
            ]
        }

        # Empty conversation
        self.empty_result: dict[str, list] = {
            'messages': []
        }

        # Minimal conversation
        self.minimal_result = {
            'messages': [
                {'role': 'user', 'content': 'Hi!'}
            ]
        }

    def tearDown(self) -> None:
        self.mock_tokenizer.stop()

    def test_tokens_used_score_typical(self) -> None:
        expected_token_count = 8  # Count of words in typical_result
        self.assertEqual(tokens_used_score(self.typical_result), expected_token_count)

    def test_tokens_used_score_empty(self) -> None:
        expected_token_count = 0
        self.assertEqual(tokens_used_score(self.empty_result), expected_token_count)

    def test_tokens_used_score_minimal(self) -> None:
        expected_token_count = 1  # Count of words in minimal_result
        self.assertEqual(tokens_used_score(self.minimal_result), expected_token_count)


class TestAgilityScore(unittest.TestCase):
    def setUp(self) -> None:
        # Standard llmcoder result with improving analyzer results
        self.standard_result = {
            'analyzer_results': [
                {'analyzer1': {'score': 1}, 'analyzer2': {'score': 2}},
                {'analyzer1': {'score': 3}, 'analyzer2': {'score': 4}},
                {'analyzer1': {'score': 5}, 'analyzer2': {'score': 6}}
            ]
        }

        # Constant analyzer results
        self.constant_result = {
            'analyzer_results': [
                {'analyzer1': {'score': 1}, 'analyzer2': {'score': 2}},
                {'analyzer1': {'score': 1}, 'analyzer2': {'score': 2}},
                {'analyzer1': {'score': 1}, 'analyzer2': {'score': 2}}
            ]
        }

        # Degrading analyzer results
        self.degrading_result = {
            'analyzer_results': [
                {'analyzer1': {'score': 5}, 'analyzer2': {'score': 6}},
                {'analyzer1': {'score': 3}, 'analyzer2': {'score': 4}},
                {'analyzer1': {'score': 1}, 'analyzer2': {'score': 2}}
            ]
        }

    def test_agility_score_standard(self) -> None:
        # Test with default length_scale
        score = agility_score(self.standard_result)
        self.assertTrue(isinstance(score, float))  # Checking if the output is a float
        self.assertGreater(score, 0)

    def test_agility_score_varying_length_scale(self) -> None:
        # Test with a different length_scale value
        score = agility_score(self.standard_result, length_scale=0.5)
        self.assertTrue(isinstance(score, float))
        self.assertGreater(score, 0)

    def test_agility_score_constant_results(self) -> None:
        # Test with constant analyzer results
        score = agility_score(self.constant_result)
        self.assertEqual(score, 0)  # Expecting no change in agility score for constant results

    def test_agility_score_degrading_results(self) -> None:
        # Test with degrading analyzer results
        score = agility_score(self.degrading_result)
        self.assertLess(score, 0)


class TestTimeScore(unittest.TestCase):
    def setUp(self) -> None:
        # Standard llmcoder result with a normal time value
        self.standard_result = {
            'time': 123.45
        }

        # Zero time value
        self.zero_time_result = {
            'time': 0
        }

    def test_time_score_standard(self) -> None:
        self.assertEqual(time_score(self.standard_result), 123.45)

    def test_time_score_zero_time(self) -> None:
        self.assertEqual(time_score(self.zero_time_result), 0)


class TestAllAnalyzersPassedScore(unittest.TestCase):
    def setUp(self) -> None:
        # All analyzers passed
        self.all_passed_directly_results_history = {
            'analyzer_results': [
                {
                    "mypy_analyzer_v1": {
                        "type": "critical",
                        "score": -1,
                        "pass": True,
                        "message": "Everything fine"
                    },
                    "signature_analyzer_v1": {
                        "pass": True,
                        "type": "info",
                        "score": 0,
                        "message": "All functions and classes in your completion are called correctly (their signatures match with the documentation)."
                    },
                    "gpt_score_analyzer_v1": {
                        "type": "score",
                        "score": 9.211558703193813,
                        "pass": True,
                        "message": ""
                    }
                }
            ]
        }

        # All analyzers passed in second step
        self.all_passed_directly_results_history = {
            'analyzer_results': [
                {
                    "mypy_analyzer_v1": {
                        "type": "critical",
                        "score": -1,
                        "pass": False,
                        "message": "Some error"
                    },
                    "signature_analyzer_v1": {
                        "pass": True,
                        "type": "info",
                        "score": 0,
                        "message": "All functions and classes in your completion are called correctly (their signatures match with the documentation)."
                    },
                    "gpt_score_analyzer_v1": {
                        "type": "score",
                        "score": 9.211558703193813,
                        "pass": True,
                        "message": ""
                    }
                }, {
                    "mypy_analyzer_v1": {
                        "type": "critical",
                        "score": -1,
                        "pass": True,
                        "message": "Now everything is fine"
                    },
                    "signature_analyzer_v1": {
                        "pass": True,
                        "type": "info",
                        "score": 0,
                        "message": "All functions and classes in your completion are called correctly (their signatures match with the documentation)."
                    },
                    "gpt_score_analyzer_v1": {
                        "type": "score",
                        "score": 9.211558703193813,
                        "pass": True,
                        "message": ""
                    }
                }
            ]
        }

        # Some analyzers failed
        self.some_failed_results_history = {
            'analyzer_results': [
                {
                    "mypy_analyzer_v1": {
                        "type": "critical",
                        "score": -1,
                        "pass": False,
                        "message": "Some error"
                    },
                    "signature_analyzer_v1": {
                        "pass": True,
                        "type": "info",
                        "score": 0,
                        "message": "All functions and classes in your completion are called correctly (their signatures match with the documentation)."
                    },
                    "gpt_score_analyzer_v1": {
                        "type": "score",
                        "score": 9.211558703193813,
                        "pass": True,
                        "message": ""
                    }
                }
            ]
        }

    def test_all_analyzers_passed_score_all_passed(self) -> None:
        self.assertTrue(all_analyzers_passed_score(self.all_passed_directly_results_history))

    def test_all_analyzers_passed_score_all_passed_second_step(self) -> None:
        self.assertTrue(all_analyzers_passed_score(self.all_passed_directly_results_history))

    def test_all_analyzers_passed_score_some_failed(self) -> None:
        self.assertFalse(all_analyzers_passed_score(self.some_failed_results_history))
