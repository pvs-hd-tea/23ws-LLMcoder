# Generated with GPT-4 under supervision

import os
import unittest
from unittest.mock import MagicMock, patch

import tiktoken

from llmcoder.llmcoder import LLMCoder
from llmcoder.utils import get_conversations_dir


# Define the mock response classes
class MockMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class MockChoice:
    def __init__(self, message: MockMessage) -> None:
        self.message = message


class MockCompletionResponse:
    def __init__(self, choices: list[MockChoice]) -> None:
        self.choices = choices


# Define the helper function
def create_mock_openai_response(content: str) -> MockCompletionResponse:
    mock_message = MockMessage(content)
    mock_choice = MockChoice(mock_message)
    return MockCompletionResponse([mock_choice])


class TestLLMCoder(unittest.TestCase):
    def setUp(self) -> None:
        # FIXME: Use a mock instead of a real file. This currently fails because the get_openai_key is not patched correctly.
        self.key_file_path = os.path.join(os.path.dirname(__file__), '..', 'key.txt')
        if not os.path.isfile(self.key_file_path):
            with open(self.key_file_path, "w") as f:
                f.write("sk-mock_key")

        self.enc = tiktoken.get_encoding('p50k_base')

    def tearDown(self) -> None:
        if os.path.isfile(self.key_file_path):
            os.remove(self.key_file_path)
