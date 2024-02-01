import unittest

from llmcoder.conversation import Conversation, PriorityQueue


class TestPriorityQueue(unittest.TestCase):
    def setUp(self) -> None:
        # Mock Conversation objects for testing
        self.conv1 = Conversation(10, [], [])
        self.conv2 = Conversation(20, [], [])
        self.conv3 = Conversation(30, [], [])

    def test_init_empty(self) -> None:
        """Test initializing an empty priority queue."""
        pq = PriorityQueue()
        self.assertEqual(len(pq), 0)

    def test_init_with_single_conversation(self) -> None:
        """Test initializing the priority queue with a single conversation."""
        pq = PriorityQueue(self.conv1)
        self.assertEqual(len(pq), 1)

    def test_init_with_multiple_conversations(self) -> None:
        """Test initializing the priority queue with multiple conversations."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        self.assertEqual(len(pq), 3)

    def test_push(self) -> None:
        """Test pushing a conversation onto the priority queue."""
        pq = PriorityQueue()
        pq.push(self.conv1)
        self.assertEqual(len(pq), 1)

    def test_pop(self) -> None:
        """Test popping the highest priority (lowest score) conversation."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        highest_priority_conv = pq.pop()
        self.assertEqual(highest_priority_conv.score, 30)  # Since score is inverted twice
        self.assertEqual(len(pq), 2)

    def test_pop_keep(self) -> None:
        """Test popping the highest priority conversation without removing it."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        highest_priority_conv = pq.pop(keep=True)
        self.assertEqual(highest_priority_conv.score, 30)  # Since score is inverted twice
        self.assertEqual(len(pq), 3)  # Queue length should remain unchanged

    def test_clear(self) -> None:
        """Test clearing the priority queue."""
        pq = PriorityQueue([self.conv1, self.conv2])
        pq.clear()
        self.assertEqual(len(pq), 0)

    def test_iteration(self) -> None:
        """Test iterating over the priority queue."""
        pq = PriorityQueue([self.conv1, self.conv2])
        scores = [-conv.score for conv in pq]  # Scores are inverted in the queue
        self.assertIn(10, scores)
        self.assertIn(20, scores)
