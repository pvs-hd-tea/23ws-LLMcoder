# Generated with GPT-4 under supervision

import unittest
from unittest.mock import MagicMock, patch

from llmcoder.eval.metrics.extrinsic import bleu_score, levenshtein_distance_score, sequence_matcher_score, trf_similarity_score


class TestLevenshteinDistanceScore(unittest.TestCase):

    def test_typical_strings(self) -> None:
        ground_truth = "kitten"
        llmcoder_result = "sitting"
        expected_distance = 3  # As per Levenshtein distance calculation
        self.assertEqual(levenshtein_distance_score(ground_truth, llmcoder_result), expected_distance)

    def test_empty_strings(self) -> None:
        ground_truth = ""
        llmcoder_result = ""
        expected_distance = 0
        self.assertEqual(levenshtein_distance_score(ground_truth, llmcoder_result), expected_distance)

    def test_one_empty_string(self) -> None:
        ground_truth = ""
        llmcoder_result = "non-empty"
        expected_distance = len(llmcoder_result)
        self.assertEqual(levenshtein_distance_score(ground_truth, llmcoder_result), expected_distance)

    def test_identical_strings(self) -> None:
        ground_truth = "identical"
        llmcoder_result = "identical"
        expected_distance = 0
        self.assertEqual(levenshtein_distance_score(ground_truth, llmcoder_result), expected_distance)

    def test_llmcoder_result_as_dict(self) -> None:
        ground_truth = "kitten"
        llmcoder_result = {'messages': [{'content': 'sitting'}]}
        expected_distance = 3
        self.assertEqual(levenshtein_distance_score(ground_truth, llmcoder_result), expected_distance)


class TestBleuScore(unittest.TestCase):

    def test_standard_reference_and_candidate(self) -> None:
        ground_truth = "the quick brown fox jumps over the lazy dog"
        llmcoder_result = "a quick brown fox jumps over the lazy dog"
        # Expect some BLEU score, not 0 and not 1
        score = bleu_score(ground_truth, llmcoder_result)
        self.assertTrue(0 < score < 1)

    def test_multiple_references(self) -> None:
        ground_truth = ["the quick brown fox jumps over the lazy dog", "a fast brown fox leaps over a lazy dog"]
        llmcoder_result = "a quick brown fox jumps over the lazy dog"
        # Expect some BLEU score, not 0 and not 1
        score = bleu_score(ground_truth, llmcoder_result)
        self.assertTrue(0 < score < 1)

    def test_empty_reference(self) -> None:
        ground_truth = ""
        llmcoder_result = "some content"
        expected_score = 0  # BLEU score is 0 when reference is empty
        self.assertEqual(bleu_score(ground_truth, llmcoder_result), expected_score)

    def test_empty_candidate(self) -> None:
        ground_truth = "the quick brown fox jumps over the lazy dog"
        llmcoder_result = ""
        expected_score = 0  # BLEU score is 0 when candidate is empty
        self.assertEqual(bleu_score(ground_truth, llmcoder_result), expected_score)

    def test_llmcoder_result_as_dict(self) -> None:
        ground_truth = "the quick brown fox jumps over the lazy dog"
        llmcoder_result = {'messages': [{'content': 'a quick brown fox jumps over the lazy dog'}]}
        # Expect some BLEU score, not 0 and not 1
        score = bleu_score(ground_truth, llmcoder_result)
        self.assertTrue(0 < score < 1)


class TestTrfSimilarityScore(unittest.TestCase):

    def setUp(self) -> None:
        # Mock SentenceTransformer and its encode method
        self.patcher = patch('sentence_transformers.SentenceTransformer')
        self.mock_model_class = self.patcher.start()
        self.mock_model_instance = MagicMock()
        self.mock_model_instance.encode.side_effect = lambda text: text
        self.mock_model_class.return_value = self.mock_model_instance

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_standard_strings(self) -> None:
        ground_truth = "This is a test."
        llmcoder_result = "This is a test."
        # Expect high similarity for identical strings
        score = trf_similarity_score(ground_truth, llmcoder_result)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_empty_strings(self) -> None:
        ground_truth = ""
        llmcoder_result = ""
        # Similarity might be undefined for empty strings; adjust as per your implementation
        score = trf_similarity_score(ground_truth, llmcoder_result)
        self.assertTrue(isinstance(score, float))

    def test_identical_strings(self) -> None:
        ground_truth = "identical"
        llmcoder_result = "identical"
        score = trf_similarity_score(ground_truth, llmcoder_result)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_llmcoder_result_as_dict(self) -> None:
        ground_truth = "This is a test."
        llmcoder_result = {'messages': [{'content': 'This is a test.'}]}
        score = trf_similarity_score(ground_truth, llmcoder_result)
        self.assertAlmostEqual(score, 1.0, places=4)


class TestSequenceMatcherScore(unittest.TestCase):

    def test_standard_strings(self) -> None:
        ground_truth = "Hello World"
        llmcoder_result = "Halo World"
        # Expect some similarity score, not 0 and not 1
        score = sequence_matcher_score(ground_truth, llmcoder_result)
        self.assertTrue(0 < score < 1)

    def test_empty_strings(self) -> None:
        ground_truth = ""
        llmcoder_result = ""
        expected_score = 1.0  # Similarity is 1 for two empty strings
        self.assertEqual(sequence_matcher_score(ground_truth, llmcoder_result), expected_score)

    def test_one_empty_string(self) -> None:
        ground_truth = ""
        llmcoder_result = "non-empty"
        expected_score = 0.0  # Similarity is 0 when one string is empty
        self.assertEqual(sequence_matcher_score(ground_truth, llmcoder_result), expected_score)

    def test_identical_strings(self) -> None:
        ground_truth = "identical"
        llmcoder_result = "identical"
        expected_score = 1.0  # Similarity is 1 for identical strings
        self.assertEqual(sequence_matcher_score(ground_truth, llmcoder_result), expected_score)

    def test_llmcoder_result_as_dict(self) -> None:
        ground_truth = "Hello World"
        llmcoder_result = {'messages': [{'content': 'Halo World'}]}
        # Expect some similarity score, not 0 and not 1
        score = sequence_matcher_score(ground_truth, llmcoder_result)
        self.assertTrue(0 < score < 1)
