import unittest

import numpy as np

from llmcoder.conversation import Conversation, PriorityQueue, softmax


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

    def test_push_single(self) -> None:
        """Test pushing a conversation onto the priority queue."""
        pq = PriorityQueue()
        pq.push(self.conv1)
        self.assertEqual(len(pq), 1)

    def test_push_multiple(self) -> None:
        """Test pushing multiple conversations onto the priority queue."""
        pq = PriorityQueue()
        pq.push([self.conv2, self.conv1, self.conv3])  # Input not sorted
        self.assertEqual(len(pq), 3)
        self.assertGreaterEqual(pq.queue[0].score, pq.queue[1].score)
        self.assertGreaterEqual(pq.queue[1].score, pq.queue[2].score)

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

    def test_pop_temperature(self) -> None:
        """Test popping a conversation using temperature."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        highest_priority_conv = pq.pop(temperature=0.5)
        self.assertIsInstance(highest_priority_conv, Conversation)

    def test_sample(self) -> None:
        """Test sampling a conversation from the priority queue using softmax."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        index = pq.sample(temperature=0.5)
        self.assertIsInstance(index, int)

    def test_sample_uniform(self) -> None:
        """Test sampling a conversation uniformly from the priority queue."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        index = pq.sample(temperature=np.inf)
        self.assertIsInstance(index, int)

    def test_sample_temperature_zero(self) -> None:
        """Test sampling a conversation from the priority queue when temperature is 0."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        index = pq.sample(temperature=0)
        self.assertTrue(index == 0)

    def test_get_probabilities(self) -> None:
        """Test getting the probabilities of the conversations in the priority queue."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        probabilities = pq.get_probabilities(temperature=0.5)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertAlmostEqual(sum(probabilities), 1)

    def test_get_probabilities_temperature_zero(self) -> None:
        """Test getting the probabilities of the conversations in the priority queue when temperature is 0."""
        pq = PriorityQueue([self.conv1, self.conv2, self.conv3])
        probabilities = pq.get_probabilities(temperature=0)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertTrue(sum(probabilities) == 1)
        self.assertTrue(probabilities[0] == 1)

    def test_clear(self) -> None:
        """Test clearing the priority queue."""
        pq = PriorityQueue([self.conv1, self.conv2])
        pq.clear()
        self.assertEqual(len(pq), 0)

    def test_iteration(self) -> None:
        """Test iterating over the priority queue."""
        pq = PriorityQueue([self.conv1, self.conv2])
        scores = [conv.score for conv in pq]  # Scores are inverted in the queue
        self.assertIn(10, scores)
        self.assertIn(20, scores)


def test_softmax() -> None:
    """Test the softmax function."""
    scores = [1, 2, 3]
    probabilities = softmax(scores, temperature=0.5)
    assert sum(probabilities) == 1
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert probabilities[0] < 1 / 3
    assert probabilities[2] > 1 / 3
