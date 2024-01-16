import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml
from dynaconf import Dynaconf

from llmcoder.eval.evaluate import Evaluation  # Replace with your module name


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary file for the configuration
        self.temp_config_file, self.temp_config_path = tempfile.mkstemp(suffix='.yaml', prefix='pytest_config_')

        self.config_file_name = os.path.basename(self.temp_config_path)

        # Configuration data to write to the file
        config_data = {
            'analyzers': ["mypy_analyzer_v1", "signature_analyzer_v1", "gpt_score_analyzer_v1"],
            'model_first': "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d",
            'model_feedback': "gpt-3.5-turbo",
            'feedback_variant': "coworker",
            'system_prompt': "2023-11-15_GPT-Builder.txt",
            'dataset': "pytest_dataset",
            'max_iter': 5,
            'log_conversation': True,
            'scores': [
                "extrinsic.levenshtein_distance_score",
                "extrinsic.bleu_score",
                "extrinsic.trf_similarity_score",
                "extrinsic.sequence_matcher_score",
                "extrinsic.gpt_reviewer_score",
                "intrinsic.loops_required_score",
                "intrinsic.tokens_used_score",
                "intrinsic.agility_score",
                "intrinsic.time_score",
                "intrinsic.all_analyzers_passed_score"
            ],
            'n_choices': 3,
            'n_procs': 3
        }

        # Write the configuration data to the temporary file
        with open(self.temp_config_path, 'w') as file:
            yaml.dump(config_data, file)

        # Mocking os.path.exists to always return True for the temp file
        self.patcher_exists = patch('os.path.exists', return_value=True)
        self.patcher_exists.start()

        # Mocking os.listdir to return only our temp config file
        self.patcher_listdir = patch('os.listdir', return_value=[self.config_file_name])
        self.patcher_listdir.start()

    def tearDown(self) -> None:
        # Stop the patchers
        self.patcher_exists.stop()
        self.patcher_listdir.stop()

        # Close and remove the temporary configuration file
        os.close(self.temp_config_file)
        os.remove(self.temp_config_path)

    @patch('llmcoder.eval.evaluate.get_config_dir')
    def test_init_no_configs(self, mock_get_config_dir: MagicMock) -> None:
        mock_get_config_dir.return_value = os.path.dirname(self.temp_config_path)
        evaluation = Evaluation()
        self.assertEqual(len(evaluation.configs), 1)

    @patch('os.path.exists')
    def test_init_single_config(self, mock_exists: MagicMock) -> None:
        mock_exists.return_value = True
        config = Dynaconf(settings_files=[self.temp_config_path])
        evaluation = Evaluation(config)
        self.assertEqual(len(evaluation.configs), 1)

    @patch('llmcoder.eval.evaluate.get_config_dir')
    @patch('os.path.exists')
    def test_init_multiple_configs(self, mock_exists: MagicMock, mock_get_config_dir: MagicMock) -> None:
        mock_exists.return_value = True
        mock_get_config_dir.return_value = os.path.dirname(self.temp_config_path)
        configs = [Dynaconf(settings_files=[self.temp_config_path]), Dynaconf(settings_files=[self.temp_config_path])]
        evaluation = Evaluation(configs)
        self.assertEqual(len(evaluation.configs), 2)

    @patch('llmcoder.eval.evaluate.get_config_dir')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_init_invalid_dataset_path(self, mock_listdir: MagicMock, mock_exists: MagicMock, mock_get_config_dir: MagicMock) -> None:
        mock_get_config_dir.return_value = os.path.dirname(self.temp_config_path)
        mock_listdir.return_value = [os.path.basename(self.temp_config_path)]
        mock_exists.side_effect = lambda path: path != self.temp_config_path
        config = Dynaconf(settings_files=[self.temp_config_path])
        with self.assertRaises(TypeError):  # Happens in the check_config function because dynaconf.get('analyzers') is None instead of a string
            Evaluation(config)
