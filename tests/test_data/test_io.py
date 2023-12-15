import json
import os
import shutil
import unittest
from unittest.mock import mock_open, patch

from llmcoder.data.io import construct_eval_conversation, dump_results_to_json, read_data_from_conversations_file, read_data_from_pairs, read_results_from_json


class TestReadDataFromPairs(unittest.TestCase):
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")

    def setUp(self) -> None:
        # Create a test directory with pairs
        for i in range(1, 3):
            os.makedirs(os.path.join(self.test_dir, f"pair{i}"), exist_ok=True)
            for split in ["input", "output"]:
                with open(os.path.join(self.test_dir, f"pair{i}", f"{split}.txt"), "w") as f:
                    f.write(f"{split}{i}")

    def tearDown(self) -> None:
        # Remove the test directory
        shutil.rmtree(self.test_dir)
        pass

    def test_read_data_from_pairs(self) -> None:
        # Assuming a test directory with pairs exists
        expected_pairs = [('input1', 'output1'), ('input2', 'output2')]
        actual_pairs = read_data_from_pairs(self.test_dir)
        self.assertEqual(set(actual_pairs), set(expected_pairs))


class TestReadDataFromConversationsFile(unittest.TestCase):

    def test_read_data_from_conversations_file(self) -> None:
        test_file = "test_conversations.jsonl"
        # Create the test file
        with open(test_file, "w") as f:
            f.write(json.dumps({"messages": [{"role": "system", "content": "test_system"},
                                             {"role": "user", "content": "test_user"},
                                             {"role": "assistant", "content": "test_assistant"}]}))
        expected_pairs = [("test_user", "test_assistant")]
        actual_pairs = read_data_from_conversations_file(test_file)
        self.assertEqual(actual_pairs, expected_pairs)


class TestConstructEvalConversation(unittest.TestCase):

    def test_construct_eval_conversation(self) -> None:
        pairs = [('input1', 'output1'), ('input2', 'output2')]
        system_prompt = "System prompt"
        expected_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "input1"},
            {"role": "assistant", "content": "output1"},
            {"role": "user", "content": "input2"},
            {"role": "assistant", "content": "output2"}
        ]
        actual_conversation = construct_eval_conversation(pairs, system_prompt)
        self.assertEqual(actual_conversation, expected_conversation)


class TestJsonOperations(unittest.TestCase):

    def test_dump_results_to_json(self) -> None:
        results = {"key": "value"}
        with patch("builtins.open", mock_open()) as mock_file:
            dump_results_to_json(results, "test_output.json")
            mock_file.assert_called_once_with("test_output.json", 'w')

    def test_read_results_from_json(self) -> None:
        data = {"key": "value"}
        with patch("builtins.open", mock_open(read_data=json.dumps(data))):
            result = read_results_from_json("test_input.json")
            self.assertEqual(result, data)
