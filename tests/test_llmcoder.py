import os
import unittest
from unittest.mock import MagicMock, patch

from llmcoder.LLMCoder import LLMCoder  # Replace with the actual module name where LLMCoder is defined
from llmcoder.utils import get_conversations_dir


class TestLLMCoder(unittest.TestCase):
    @patch('llmcoder.utils.get_openai_key', return_value="mock_api_key")
    @patch('llmcoder.utils.get_conversations_dir', return_value="/mock/conversations/dir")
    @patch('llmcoder.utils.get_system_prompt', return_value="mock_system_prompt")
    @patch('llmcoder.LLMCoder.LLMCoder._create_conversation_file', return_value=None)
    @patch('openai.OpenAI')
    def test_init_default_parameters(self, mock_openai: MagicMock, mock_system_prompt: MagicMock, mock_conversations_dir: MagicMock, mock_openai_key: MagicMock, mock_create_conversation_file: MagicMock) -> None:
        llmcoder = LLMCoder()
        self.assertEqual(llmcoder.analyzers, [])
        self.assertEqual(llmcoder.model_first, "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d")
        self.assertEqual(llmcoder.model_feedback, "gpt-3.5-turbo")
        self.assertEqual(llmcoder.feedback_variant, "separate")
        self.assertEqual(llmcoder.max_iter, 10)

        # Check that a conversation file is created
        self.assertEqual(llmcoder.conversation_file, None)

    def test_create_conversation_file(self) -> None:
        conversations_dir = get_conversations_dir()
        conversation_file = LLMCoder._create_conversation_file()
        self.assertEqual(os.path.dirname(conversation_file), conversations_dir)

        # Check the extension
        self.assertEqual(os.path.splitext(conversation_file)[1], ".jsonl")

    @patch('openai.OpenAI')
    @patch('os.path')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="mock_data")
    @patch('os.makedirs')  # Mock os.makedirs
    @patch('json.dumps')
    def test_add_message(self, mock_openai: MagicMock, mock_path: MagicMock, mock_open: MagicMock, mock_os_makedirs: MagicMock, mock_json_dumps: MagicMock) -> None:
        llmcoder = LLMCoder(log_conversation=True)

        # Check if the system prompt is added
        self.assertEqual(len(llmcoder.messages), 1)
        self.assertEqual(llmcoder.messages[0], {"role": "system", "content": "mock_data"})

        # Mocking a response from OpenAI client
        mock_response = {"choices": [{"message": {"content": "mock_response"}}]}
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Test adding a user message
        llmcoder._add_message("user", message="Hello, world!")
        self.assertEqual(len(llmcoder.messages), 2)
        self.assertEqual(llmcoder.messages[1], {"role": "user", "content": "Hello, world!"})

        # Test adding an assistant message without predefined content
        llmcoder._add_message("assistant")
        self.assertEqual(len(llmcoder.messages), 3)
        self.assertTrue("content" in llmcoder.messages[2])
        self.assertTrue(llmcoder.messages[2]["content"].startswith("mock_response"))

    @patch('openai.OpenAI')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="mock_data")
    @patch('json.dumps')
    def test_complete_first(self, mock_openai: MagicMock, mock_open: MagicMock, mock_json_dumps: MagicMock) -> None:
        llmcoder = LLMCoder(model_first="ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d")

        # Mock the OpenAI client response
        mock_response = {"choices": [{"message": {"content": "mock_completed_code"}}]}
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Call complete_first
        code = "print('Hello, World!')"
        _ = llmcoder.complete_first(code)

        # Check if iterations is set to 0
        self.assertEqual(llmcoder.iterations, 0)

        # Check if the completion is correct
        self.assertTrue("content" in llmcoder.messages[-1])

    @patch('openai.OpenAI')
    @patch('llmcoder.analyze.Analyzer')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="mock_data")
    @patch('json.dumps')
    def test_feedback_step(self, mock_openai: MagicMock, mock_analyzer_class: MagicMock, mock_open: MagicMock, mock_json_dumps: MagicMock) -> None:
        llmcoder = LLMCoder()

        # Mock the OpenAI client response
        mock_response = {"choices": [{"message": {"content": "mock_completed_code"}}]}
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Create two mock analyzers
        mock_analyzer1 = MagicMock()
        mock_analyzer1.analyze.return_value = {"pass": True, "message": ""}
        mock_analyzer2 = MagicMock()
        mock_analyzer2.analyze.return_value = {"pass": False, "message": "Error message"}

        # Set up the state of the LLMCoder object
        llmcoder.messages = [{'content': 'print("Hello, World!")'}, {'content': 'print("Goodbye, World!")'}]
        llmcoder.feedback_variant = 'separate'
        llmcoder.analyzers = [mock_analyzer1, mock_analyzer2]

        # Call feedback_step
        result = llmcoder.feedback_step()

        # Check if the result is correct
        self.assertFalse(result)

        # Check if the state of the LLMCoder object is correct
        self.assertEqual(llmcoder.iterations, 1)
        self.assertEqual(llmcoder.messages[-2]['content'], 'Error message')

    @patch('openai.OpenAI')
    @patch('llmcoder.analyze.Analyzer')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="mock_data")
    @patch('json.dumps')
    def test_complete(self, mock_openai: MagicMock, mock_analyzer_class: MagicMock, mock_open: MagicMock, mock_json_dumps: MagicMock) -> None:
        llmcoder = LLMCoder(max_iter=2)

        # Mock the OpenAI client response
        mock_response = {"choices": [{"message": {"content": "mock_completed_code"}}]}
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Create two mock analyzers
        mock_analyzer1 = MagicMock()
        mock_analyzer1.analyze.return_value = {"pass": True, "message": ""}
        mock_analyzer2 = MagicMock()
        mock_analyzer2.analyze.return_value = {"pass": False, "message": "Error message"}

        # Set up the state of the LLMCoder object
        llmcoder.feedback_variant = 'separate'
        llmcoder.analyzers = [mock_analyzer1, mock_analyzer2]

        # Call complete
        _ = llmcoder.complete('print("Hello, World!")')

        print(llmcoder.messages)

        # Check if the state of the LLMCoder object is correct
        self.assertEqual(llmcoder.iterations, 2)
        self.assertEqual(llmcoder.messages[0]['role'], 'system')
        self.assertEqual(llmcoder.messages[1]['role'], 'user')
        self.assertEqual(llmcoder.messages[2]['role'], 'assistant')
        self.assertEqual(llmcoder.messages[3]['role'], 'user')
        self.assertEqual(llmcoder.messages[4]['role'], 'assistant')
