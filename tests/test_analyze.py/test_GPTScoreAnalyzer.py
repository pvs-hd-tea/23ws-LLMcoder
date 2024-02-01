# Generated with GPT-4 under supervision

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from openai import OpenAI

from llmcoder.analyze.gpt_score_analyzer import GPTScoreAnalyzer


class TestGPTScoreAnalyzer(unittest.TestCase):

    def setUp(self) -> None:
        # Mock the OpenAI client
        self.mock_client = MagicMock(spec=OpenAI, chat=MagicMock(completions=MagicMock(create=MagicMock())))

        # Set up the nested attributes and methods
        # Assuming 'chat' has a 'completions' attribute which in turn has a 'create' method
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    content="Score: 1\nScore: 2\nScore: 3\nScore: 4"))])

        # Create an instance of GPTScoreAnalyzer
        self.analyzer = GPTScoreAnalyzer(client=self.mock_client)

    def test_initialization(self) -> None:
        # Test default initialization
        default_analyzer = GPTScoreAnalyzer()
        self.assertIsNotNone(default_analyzer.client)
        self.assertIsNotNone(default_analyzer.scoring_prompt)
        self.assertEqual(default_analyzer.reduction, "geo")

        # Test custom initialization
        custom_analyzer = GPTScoreAnalyzer(client=self.mock_client, scoring_prompt="Custom prompt", reduction="mean")
        self.assertEqual(custom_analyzer.client, self.mock_client)
        self.assertEqual(custom_analyzer.scoring_prompt, "Custom prompt")
        self.assertEqual(custom_analyzer.reduction, "mean")

    def test_score_prompt(self) -> None:
        code_list = ["print('Hello, World!')", "x = 5"]
        expected_prompt = "Code snippet 1:\n```python\nprint('Hello, World!')\n```\n\nCode snippet 2:\n```python\nx = 5\n```\n\n"
        self.assertEqual(self.analyzer.score_prompt(code_list), expected_prompt)

    @patch('llmcoder.analyze.GPTScoreAnalyzer.GPTScoreAnalyzer.score_prompt')
    def test_score_code(self, mock_score_prompt: MagicMock) -> None:
        mock_score_prompt.return_value = "Mocked prompt"

        # Test with single code snippet
        scores = self.analyzer.score_code("print('Hello, World!')")
        self.assertTrue(isinstance(scores, np.ndarray))
        self.assertEqual(scores.shape, (1,))

    @patch('llmcoder.analyze.GPTScoreAnalyzer.GPTScoreAnalyzer.score_prompt')
    def test_score_code_multi(self, mock_score_prompt: MagicMock) -> None:
        # Mock the OpenAI client
        mock_client = MagicMock(spec=OpenAI, chat=MagicMock(completions=MagicMock(create=MagicMock())))

        # Set up the nested attributes and methods
        # Assuming 'chat' has a 'completions' attribute which in turn has a 'create' method
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    content="Code snippet 1:\nScore: 1\nScore: 2\nScore: 3\nScore: 4\n\nCode snippet 2:\nScore: 5\nScore: 6\nScore: 7\nScore: 8\n"))])

        # Create an instance of GPTScoreAnalyzer
        analyzer = GPTScoreAnalyzer(client=mock_client)

        # Test with list of code snippets
        scores = analyzer.score_code(["print('Hello, World!')", "x = 5"])
        self.assertTrue(isinstance(scores, np.ndarray))
        self.assertEqual(scores.shape, (2,))

    def test_score_code_invalid(self) -> None:
        # Mock the OpenAI client
        mock_client = MagicMock(spec=OpenAI, chat=MagicMock(completions=MagicMock(create=MagicMock())))

        # Set up the nested attributes and methods
        # Assuming 'chat' has a 'completions' attribute which in turn has a 'create' method
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    content="Code snippet 1:\nScore: 1\nScore: 2\nScore: 3\nScore: 4\n\nCode snippet 2:\nScore: 5\nScore: 6\nScore: 7\nScore: 8\n"))])

        # Create an instance of GPTScoreAnalyzer
        analyzer = GPTScoreAnalyzer(client=mock_client, reduction="invalid")

        # Test with invalid reduction method
        with self.assertRaises(ValueError):
            analyzer.reduction = "invalid"
            analyzer.score_code("print('Hello, World!')")

    def test_analyze(self) -> None:
        # Mock score_code to return a fixed value
        self.analyzer.score_code = MagicMock(return_value=np.array([1.0]))

        result = self.analyzer.analyze("input", "completion")
        expected_result = {
            "type": "score",
            "score": 1.0,
            "pass": True,
            "message": ""
        }
        self.assertEqual(result, expected_result)

        # Test with invalid reduction method
        with self.assertRaises(ValueError):
            self.analyzer.reduction = None
            self.analyzer.analyze("input", "completion")
