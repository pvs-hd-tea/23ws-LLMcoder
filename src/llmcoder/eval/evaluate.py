import importlib
import os
from datetime import datetime

import pandas as pd
from dynaconf import Dynaconf
from tqdm import tqdm

from llmcoder.data.io import dump_results_to_json, read_data_from_conversations_file
from llmcoder.LLMCoder import LLMCoder  # Import your LLMCoder class
from llmcoder.utils import get_data_dir


class Evaluation:
    def __init__(self, config: Dynaconf):
        """
        Initialize the Evaluation with a Dynaconf configuration.

        Parameters
        ----------
        config : Dynaconf
            The configuration object from Dynaconf.
        """
        self.config = config
        self.data: list[tuple[str, str]] | None = None
        self.results: dict | None = None

    def predict(self, store: bool = False, verbose: bool = False) -> dict:
        """
        Run the LLMCoder on the provided files and write the results to the database.

        Returns
        -------
        dict
            The results from the evaluation.
        """
        self.data = self._get_data()
        self.inputs = {i: input for i, (input, _) in enumerate(self.data)}
        self.targets = {i: target for i, (_, target) in enumerate(self.data)}
        self.results = self._run(self.inputs, verbose=verbose)

        if store:
            self._write_results(self.results)

        return self.results

    def run(self, verbose: bool = False) -> dict:
        """
        Run the evaluation end to end (reading inputs from the database and writing results back)
        """
        results = self.predict(store=True, verbose=verbose)
        self.analyze(results, store=True, verbose=verbose)

        return self.analysis_results

    def analyze(self, results: dict, store: bool = False, verbose: bool = False) -> dict:
        """
        Analyze the results from the database given the configuration and store it back in the database.
        """
        intrinsic_score_functions = [getattr(importlib.import_module(f'llmcoder.eval.metrics.{score.split(".")[0]}'), score.split(".")[1]) for score in self.config.scores if score.split(".")[0] == 'intrinsic']
        extrinsic_score_functions = [getattr(importlib.import_module(f'llmcoder.eval.metrics.{score.split(".")[0]}'), score.split(".")[1]) for score in self.config.scores if score.split(".")[0] == 'extrinsic']

        self.analysis_results: dict[str, dict] = {}

        for results_id, result in tqdm(results.items(), desc='Analysis', total=len(results), disable=not verbose):
            self.analysis_results[results_id] = {}
            for f in intrinsic_score_functions:
                self.analysis_results[results_id][f.__name__] = f(result['messages'][-1]['content'], self.targets[results_id])
            for f in extrinsic_score_functions:
                self.analysis_results[results_id][f.__name__] = f(result['messages'][-1]['content'], self.targets[results_id])

        if store:
            self._write_analysis_results(self.analysis_results)

        return self.analysis_results

    def _get_results(self) -> list[dict]:
        """
        Get the results from the database.

        Returns
        -------
        List[dict]
            A list of results from the database.
        """
        raise NotImplementedError

    def _get_data(self) -> list[tuple[str, str]]:
        """
        Get the inputs to run the LLMCoder on.

        Returns
        -------
        list[tuple[str, str]]
            A list of inputs to run the LLMCoder on and outputs to compare the results to.
        """
        return read_data_from_conversations_file(os.path.join(get_data_dir(self.config.get('dataset')), 'conversations.jsonl'))

    def _run(self, inputs: dict, verbose: bool = False) -> dict:
        """
        Run the LLMCoder on the provided files and return the results.

        Parameters
        ----------
        inputs : List[str]
            A list of inputs to complete with the LLMCoder.
        verbose : bool, optional
            Whether to print the results to the console, by default False
        """

        results: dict[str, dict] = {}

        # Run the LLMCoder on each input
        for input_id, input in tqdm(inputs.items(), desc='Evaluation', total=len(inputs), disable=not verbose):
            # Initialize the LLMCoder with the configuration
            llmcoder = LLMCoder(
                analyzers=self.config.get('analyzers'),
                model_first=self.config.get('model_first'),
                model_feedback=self.config.get('model_feedback'),
                feedback_variant=self.config.get('feedback_variant'),
                system_prompt=self.config.get('system_prompt'),
                max_iter=self.config.get('max_iter'),
                log_conversation=self.config.get('log_conversation')
            )

            # Get the completion
            _ = llmcoder.complete(input)

            # Add the results to the results list
            results[input_id] = {}
            results[input_id]['messages'] = llmcoder.messages
            results[input_id]['analyzer_results'] = llmcoder.analyzer_pass_history

        return results

    def _write_results(self, results: dict) -> None:
        """
        Write the results back to the database.

        Parameters
        ----------
        results : dict
            The results to write back to the database.
        """
        dump_results_to_json(results, os.path.join(get_data_dir(self.config.get('dataset')), f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'))

    def _write_analysis_results(self, analysis_results: dict) -> None:
        """
        Write the analysis results back to the database.

        Parameters
        ----------
        analysis_results : dict
            The analysis results to write back to the database.
        """
        # Create a dataframe from the analysis results
        df = pd.DataFrame.from_dict(analysis_results, orient='index')

        # Write the dataframe to the database
        config_name = os.path.split(self.config.settings_file_for_dynaconf[0])[-1].split('.')[0]
        datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(os.path.join(get_data_dir(self.config.get('dataset')), f'results_{config_name}_{datetime_now}.csv'))
