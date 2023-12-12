import unittest

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.analyze.GPTScoreAnalyzer import GPTScoreAnalyzer
from llmcoder.analyze.MypyAnalyzer import MypyAnalyzer
from llmcoder.analyze.SignatureAnalyzer import SignatureAnalyzer


class TestAnalyzerFactory(unittest.TestCase):

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
