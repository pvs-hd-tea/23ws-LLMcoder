import unittest

from llmcoder.conversation import Conversation


class TestConversation(unittest.TestCase):
    def test_initialization(self) -> None:
        """Test that the Conversation object is initialized with the correct attributes."""
        messages = [{'id': '1', 'content': 'Hello'}]
        analyses = [{'mypy_analyzer_v1': {'score': 0.9, 'label': True}}]
        path = ['start', 'middle', 'end']
        conv = Conversation(10, messages, analyses, path)  # type: ignore
        self.assertEqual(conv.score, 10)
        self.assertEqual(conv.messages, messages)
        self.assertEqual(conv.analyses, analyses)
        self.assertEqual(conv.path, path)

    def test_copy(self) -> None:
        """Test copying a conversation object."""
        conv = Conversation(0, [{'id': '1', 'content': 'Hello'}])
        conv_copy = conv.copy()
        self.assertEqual(conv.score, conv_copy.score)
        self.assertEqual(conv.messages, conv_copy.messages)
        self.assertEqual(conv.analyses, conv_copy.analyses)
        self.assertNotEqual(id(conv), id(conv_copy))

    def test_add_message(self) -> None:
        """Test adding a message to the conversation."""
        conv = Conversation(0, [])
        message = {'id': '2', 'content': 'Test message'}
        conv.add_message(message)
        self.assertIn(message, conv.messages)

    def test_add_analysis(self) -> None:
        """Test adding an analysis to the conversation."""
        conv = Conversation(0, [], [])
        analysis = {'mypy_analyzer_v1': {'happy': 0.8}}
        conv.add_analysis(analysis)  # type: ignore
        self.assertIn(analysis, conv.analyses)

    def test_set_score(self) -> None:
        """Test setting the conversation score."""
        conv = Conversation(0, [], [])
        conv.set_score(5)
        self.assertEqual(conv.score, 5)

    def test_add_to_path(self) -> None:
        """Test adding a choice to the path."""
        conv = Conversation(0, [], [], [])
        choice = 'decision_point'
        conv.add_to_path(choice)
        self.assertIn(choice, conv.path)

    def test_update_passing_true(self) -> None:
        """Test updating the passing attribute of the conversation."""
        conv = Conversation(0, [], [{'mypy_analyzer_v1': {'pass': True, 'type': 'critical'}}])
        conv.update_passing()
        self.assertTrue(conv.passing)

    def test_update_passing_false(self) -> None:
        """Test updating the passing attribute of the conversation."""
        conv = Conversation(0, [], [{'mypy_analyzer_v1': {'pass': False, 'type': 'critical'}}])
        conv.update_passing()
        self.assertFalse(conv.passing)

    def test_get_last_message(self) -> None:
        """Test retrieving the last message from the conversation."""
        messages = [{'id': '1', 'content': 'First message'}, {'id': '2', 'content': 'Last message'}]
        conv = Conversation(0, messages)
        self.assertEqual(conv.get_last_message(), 'Last message')

    def test_comparison_operators(self) -> None:
        """Test the comparison operators for Conversation objects."""
        conv1 = Conversation(10, [])
        conv2 = Conversation(5, [])
        self.assertTrue(conv1 > conv2)
        self.assertTrue(conv1 >= conv2)
        self.assertFalse(conv1 < conv2)
        self.assertFalse(conv1 <= conv2)
        conv3 = Conversation(10, [])
        self.assertTrue(conv1 >= conv3)
        self.assertFalse(conv1 <= conv2)

    def test_contains(self) -> None:
        """Test the __contains__ method for Conversation objects."""
        conv1 = Conversation(10, [], path=['R'])
        conv2 = Conversation(5, [], path=['R', 1])

        self.assertTrue(conv1 in conv2)
        self.assertFalse(conv2 in conv1)
