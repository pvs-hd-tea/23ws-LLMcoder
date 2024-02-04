# Generated with GPT-4 under supervision

import os
import random
import string
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import tiktoken

from llmcoder.conversation import Conversation, PriorityQueue
from llmcoder.llmcoder import LLMCoder


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

    @patch('llmcoder.llmcoder.get_openai_key', return_value='test_key')
    @patch('llmcoder.llmcoder.get_system_prompt', return_value='Test prompt')
    @patch('llmcoder.llmcoder.get_conversations_dir', return_value='/tmp/conversations')
    @patch('llmcoder.llmcoder.os.listdir', return_value=[])
    @patch('llmcoder.llmcoder.tiktoken.get_encoding', return_value=MagicMock())
    @patch('llmcoder.llmcoder.openai.OpenAI')
    def test_initialization(self, mock_openai: MagicMock, mock_get_encoding: MagicMock, mock_listdir: MagicMock, mock_get_conversations_dir: MagicMock, mock_get_system_prompt: MagicMock, mock_get_openai_key: MagicMock) -> None:
        """Test LLMCoder initialization with default parameters."""
        coder = LLMCoder()
        self.assertEqual(coder.model_first, "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d")
        self.assertEqual(coder.feedback_variant, "coworker")
        self.assertIsNotNone(coder.conversation_file)
        self.assertEqual(coder.system_prompt, 'Test prompt')
        self.assertIsNotNone(coder.conversation_file)
        self.assertEqual(coder.verbose, True)
        self.assertEqual(coder.iterations, 0)
        self.assertEqual(coder.n_tokens_generated, 0)
        self.assertIsInstance(coder.conversations, PriorityQueue)
        self.assertEqual(coder.encoder, mock_get_encoding.return_value)

    @patch('llmcoder.llmcoder.get_system_prompt', return_value='System prompt for testing')
    @patch('llmcoder.llmcoder.get_openai_key', return_value='test_key')
    def test_reset_loop(self, mock_get_openai_key: MagicMock, mock_get_system_prompt: MagicMock) -> None:
        """Test resetting the feedback loop and its internal variables."""
        coder = LLMCoder(log_conversation=False)
        coder._reset_loop()  # Reset loop to ensure it sets up correctly
        self.assertEqual(coder.iterations, 0)
        self.assertEqual(coder.n_tokens_generated, 0)
        self.assertEqual(coder.conversations.pop(keep=True).get_last_message(), 'System prompt for testing')

    def test_get_best_completion(self) -> None:
        """Test getting the best completion from the OpenAI API response."""
        coder = LLMCoder(log_conversation=False)
        coder.conversations.pop()

        conversations = [
            Conversation(-2, [
                {'role': 'pytest', 'content': 'Test system message'},
                {'role': 'pytest', 'content': 'Test user message'}
            ], [
                {},
                {
                    "analyzer1": {"type": "critical", "pass": False},
                    "analyzer2": {"type": "critical", "pass": False},
                }
            ]),
            Conversation(-1, [
                {'role': 'pytest', 'content': 'Test system message'},
                {'role': 'pytest', 'content': 'Better Test message'}
            ], [
                {},
                {
                    "analyzer1": {"type": "critical", "pass": False},
                    "analyzer2": {"type": "critical", "pass": False},
                }
            ]),
        ]

        for conversation in conversations:
            conversation.update_passing()

        best_completion = coder._get_best_completion(conversations)

        self.assertEqual(best_completion, 'Better Test message')

    @patch('openai.OpenAI')
    def test_complete(self, mock_openai: MagicMock) -> None:
        """Test the completion of a conversation."""
        mock_openai.return_value.chat.completions.create.return_value = create_mock_openai_response('Test message')

        coder = LLMCoder(log_conversation=False)
        completion = coder.complete('Test prompt')
        self.assertEqual(completion, 'Test message')

    @patch('llmcoder.llmcoder.get_conversations_dir', return_value='/tmp/conversations')
    @patch('llmcoder.llmcoder.datetime')
    def test_create_conversation_file(self, mock_datetime: MagicMock, mock_get_conversations_dir: MagicMock) -> None:
        """Test the conversation file path creation."""
        # Setup the mock datetime to return a fixed datetime
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        expected_date_str = '2024-01-01 12:00:00'
        expected_path = f'/tmp/conversations/{expected_date_str}.jsonl'

        # Call the method under test
        file_path = LLMCoder._create_conversation_file()

        # Assert the file path matches the expected path
        self.assertEqual(file_path, expected_path)

        # Ensure the get_conversations_dir was called with create=True
        mock_get_conversations_dir.assert_called_once_with(create=True)

    def test_completion_not_in_conversations(self) -> None:
        """Test with a completion that has not appeared in any conversation."""
        coder = LLMCoder(log_conversation=False)
        # Add conversations with messages that do not match the completion being tested
        coder.conversations.push(Conversation(score=0, messages=[{"role": "user", "content": "Hello world"}]))
        coder.conversations.push(Conversation(score=0, messages=[{"role": "user", "content": "Another message"}]))
        self.assertFalse(coder._is_bad_completion("New completion"))

    def test_completion_in_conversations(self) -> None:
        """Test with a completion that has appeared in a conversation."""
        coder = LLMCoder(log_conversation=False)
        target_completion = "Repeated completion"
        # Add a conversation that includes the target completion
        coder.conversations.push(Conversation(score=0, messages=[{"role": "user", "content": "Hello world"}]))
        coder.conversations.push(Conversation(score=0, messages=[{"role": "user", "content": target_completion}]))
        self.assertTrue(coder._is_bad_completion(target_completion))

    def test_empty_conversations_list(self) -> None:
        """Test with an empty list of conversations."""
        coder = LLMCoder(log_conversation=False)
        # The conversations list is already empty as initialized in setUp
        self.assertFalse(coder._is_bad_completion("Any completion"))

    @patch('openai.OpenAI')
    def test_get_completions_for_feedback_n_equals_1(self, mock_openai: MagicMock) -> None:
        """Test getting completions for feedback."""
        mock_openai.return_value.chat.completions.create.return_value = create_mock_openai_response('Test message')

        coder = LLMCoder(log_conversation=False)

        _ = coder.conversations.pop()

        # Add a conversation to the priority queue
        coder.conversations.push(Conversation(score=10, messages=[
            {"role": "user", "content": "Test system message"},
            {"role": "user", "content": "Test user message"}
        ]))

        coder._get_completions_for(conversation=coder.conversations.pop(), n=1)

        self.assertEqual(len(coder.conversations), 1)

    @patch('openai.OpenAI')
    def test_get_completions_for_feedback_n_equals_2(self, mock_openai: MagicMock) -> None:
        """Test getting completions for feedback."""

        def create_mock_openai_response_random() -> MockCompletionResponse:
            random_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=10))
            return MockCompletionResponse(choices=[MockChoice(MockMessage(random_string + "A")), MockChoice(MockMessage(random_string + "B"))])

        mock_openai.return_value.chat.completions.create.return_value = create_mock_openai_response_random()

        coder = LLMCoder(log_conversation=False)

        _ = coder.conversations.pop()

        # Add a conversation to the priority queue
        coder.conversations.push(Conversation(score=10, messages=[
            {"role": "user", "content": "Test system message"},
            {"role": "user", "content": "Test user message"}
        ]))

        print([c.messages for c in coder.conversations.queue])

        coder._get_completions_for(conversation=coder.conversations.pop(), n=2)

        self.assertEqual(len(coder.conversations), 2)

    @patch('openai.OpenAI')
    def test_get_completions_for_feedback_n_equals_2_bad_completion(self, mock_openai: MagicMock) -> None:
        """Test getting completions for feedback with a bad completion."""
        mock_openai.return_value.chat.completions.create.return_value = create_mock_openai_response('Repeated completion')

        coder = LLMCoder(log_conversation=False)

        _ = coder.conversations.pop()

        # Add a conversation to the priority queue
        coder.conversations.push(Conversation(score=10, messages=[
            {"role": "user", "content": "Test system message"},
            {"role": "user", "content": "Test user message"}
        ]))

        coder._get_completions_for(conversation=coder.conversations.pop(), n=2)

        self.assertEqual(len(coder.conversations), 0)

    def test_run_analyzers_separate(self) -> None:
        """Test running the analyzers."""
        mock_analyzer = MagicMock(analyze=MagicMock(return_value={'pass': True, 'type': 'critical'}))

        coder = LLMCoder(log_conversation=False, feedback_variant="separate")

        coder.analyzers = {"mock_analyzer": mock_analyzer}

        code = "Test code"
        completion = "Test completion"
        results = coder._run_analyzers(code, completion)

        mock_analyzer.analyze.assert_called_once_with(code, completion)
        self.assertEqual(results, {'mock_analyzer': {'pass': True, 'type': 'critical'}})

    def test_run_analyzers_coworker(self) -> None:
        """Test running the analyzers."""
        mock_analyzer_result = {'pass': True, 'type': 'critical'}
        mock_analyzer = MagicMock(analyze=MagicMock(return_value=mock_analyzer_result))
        mock_analyzer_2 = MagicMock(analyze=MagicMock(return_value={'pass': True, 'type': 'info', 'message': 'Test message'}))

        coder = LLMCoder(log_conversation=False, feedback_variant="coworker")

        coder.analyzers = {"mock_analyzer": mock_analyzer, "mock_analyzer_2": mock_analyzer_2}

        code = "Test code"
        completion = "Test completion"
        results = coder._run_analyzers(code, completion)

        self.assertEqual(results, {'mock_analyzer': {'pass': True, 'type': 'critical'}, 'mock_analyzer_2': {'pass': True, 'type': 'info', 'message': 'Test message'}})

    def test_feedback_prompt_template(self) -> None:
        """Test the feedback prompt template."""
        coder = LLMCoder(log_conversation=False)

        result_messages = ["A", "B", "C"]
        feedback_prompt = coder._feedback_prompt_template(result_messages)

        self.assertEqual(feedback_prompt, '[INST]\nA\nB\nC\n\nFix, improve and rewrite your completion for the following code:\n[/INST]\n')

    @patch('openai.OpenAI')
    def test_step(self, mock_openai: MagicMock) -> None:
        """Test the step method."""
        mock_openai.return_value.chat.completions.create.return_value = create_mock_openai_response('Test message')

        coder = LLMCoder(log_conversation=False)

        coder.conversations.pop()

        coder.conversations.push(Conversation(score=10, messages=[
            {"role": "user", "content": "Test system message"},
            {"role": "user", "content": "Test user message"}
        ], analyses=[{"analyzer1": {"pass": True, "type": "critical"}}]))

        coder._step(code="Test code")

        self.assertEqual(len(coder.conversations), 1)
