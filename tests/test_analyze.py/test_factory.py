# Generated with GPT-4 under supervision

import os
import unittest

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.analyze.gpt_score_analyzer import GPTScoreAnalyzer
from llmcoder.analyze.mypy_analyzer import MypyAnalyzer
from llmcoder.analyze.signature_analyzer import SignatureAnalyzer


class TestAnalyzerFactory(unittest.TestCase):
    def setUp(self) -> None:
        # FIXME: Use a mock instead of a real file. This currently fails because the get_openai_key is not patched correctly.
        self.key_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'key.txt')
        if not os.path.isfile(self.key_file_path):
            with open(self.key_file_path, "w") as f:
                f.write("sk-mock_key")

    def test_create_mypy_analyzer(self) -> None:
        analyzer = AnalyzerFactory.create_analyzer("mypy_analyzer_v1")
        self.assertIsInstance(analyzer, MypyAnalyzer)

    def test_create_signature_analyzer(self) -> None:
        analyzer = AnalyzerFactory.create_analyzer("signature_analyzer_v1")
        self.assertIsInstance(analyzer, SignatureAnalyzer)

    def test_create_gpt_score_analyzer(self) -> None:
        analyzer = AnalyzerFactory.create_analyzer("gpt_score_analyzer_v1")
        self.assertIsInstance(analyzer, GPTScoreAnalyzer)

    def test_deprecation_warning(self) -> None:
        try:
            AnalyzerFactory.create_analyzer("gpt_review_analyzer_v1")
        except DeprecationWarning:
            pass
        else:
            self.fail("DeprecationWarning not raised")

    def test_invalid_analyzer_name(self) -> None:
        with self.assertRaises(ValueError):
            AnalyzerFactory.create_analyzer("invalid_analyzer_name")
