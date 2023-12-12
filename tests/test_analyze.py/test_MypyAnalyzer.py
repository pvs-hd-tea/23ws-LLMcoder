import unittest
from unittest.mock import MagicMock, patch

from llmcoder.analyze.MypyAnalyzer import MypyAnalyzer


class TestMypyAnalyzer(unittest.TestCase):

    def setUp(self) -> None:
        self.analyzer = MypyAnalyzer()

    @patch('subprocess.run')
    def test_normal_analysis(self, mock_run: MagicMock) -> None:
        # Mock subprocess.run to simulate mypy running without errors
        mock_run.return_value = MagicMock(stdout="", stderr="")

        result = self.analyzer.analyze("def hello():\n    pass\n", "print(hello())")
        expected_result = {
            "type": "critical",
            "score": 0,
            "pass": True,
            "message": "No mypy errors found."
        }
        self.assertEqual(result, expected_result)

    @patch('subprocess.run')
    def test_analysis_with_mypy_errors(self, mock_run: MagicMock) -> None:
        # Mock subprocess.run to simulate mypy running with errors
        mock_run.return_value = MagicMock(stdout="your completion:123: error: some mypy error", stderr="")

        result = self.analyzer.analyze("def hello():\n    pass\n", "print(hello")

        # Check element-wise equality
        self.assertEqual(result["type"], "critical")
        self.assertEqual(result["score"], -1)
        self.assertEqual(result["pass"], False)
        self.assertTrue("your completion" in result["message"])
        self.assertTrue("error: some mypy error" in result["message"])
