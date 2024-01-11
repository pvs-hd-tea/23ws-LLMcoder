import importlib
import io
import os
import time
from contextlib import redirect_stdout
from datetime import datetime

import pandas as pd
from dynaconf import Dynaconf
from tqdm import tqdm

from llmcoder.data.io import dump_results_to_json, dump_results_to_readable, read_data_from_conversations_file, read_results_from_json
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
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print(f'Evaluation initialized with configuration: {self.config.settings_file_for_dynaconf[0]}')
        dataset_path = os.path.abspath(os.path.join(get_data_dir(self.config.get("dataset")), 'conversations.jsonl'))
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f'Dataset not found at {dataset_path}')
        else:
            print(f'Found dataset: {self.config.get("dataset")} at {dataset_path}')

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

    def run(self, store_predictions: bool = True, store_analysis: bool = True, verbose: bool = False) -> tuple[dict, dict]:
        """
        Run the evaluation end to end (reading inputs from the database and writing results back)

        Parameters
        ----------
        store_predictions : bool, optional
            Whether to store the predictions in the database, by default True
        store_analysis : bool, optional
            Whether to store the analysis in the database, by default True
        verbose : bool, optional
            Whether to print the results to the console, by default False

        Returns
        -------
        tuple[dict, dict]
            The results from the evaluation and the analysis.
        """
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results = self.predict(store=store_predictions, verbose=verbose)
        self.analyze(results, store=store_analysis, verbose=verbose)

        return results, self.analysis_results

    def analyze(self, results: dict | str, store: bool = False, verbose: bool = False) -> dict:
        """
        Analyze the results from the database given the configuration and store it back in the database.

        Parameters
        ----------
        results : dict | str
            The results to analyze. Either a dictionary of results or the filename of a file containing the results.
        store : bool, optional
            Whether to store the analysis in the database, by default False
        verbose : bool, optional
            Whether to print the results to the console, by default False

        Returns
        -------
        dict
            The analysis results.
        """
        if isinstance(results, str):
            raise DeprecationWarning('Passing a filename to analyze is deprecated. Pass a dictionary of results instead.')
            # Try to parse the filename of the results file to get the time
            try:
                time_str = '_'.join(results.split('_')[-2:])
                time_str = time_str.split('.')[0]
                print(f'Parsed time from {results}: {time_str}')
                self.time = time_str
            except Exception:
                print(f'Could not parse time from {results}. Using default (current) time: {self.time}')

            print(f'Reading results from {results}')
            results = self._read_results(results)

        if self.data is None:
            self.data = self._get_data()
            self.inputs = {i: input for i, (input, _) in enumerate(self.data)}
            self.targets = {i: target for i, (_, target) in enumerate(self.data)}

        intrinsic_score_functions = [getattr(importlib.import_module(f'llmcoder.eval.metrics.{score.split(".")[0]}'), score.split(".")[1]) for score in self.config.scores if score.split(".")[0] == 'intrinsic']
        extrinsic_score_functions = [getattr(importlib.import_module(f'llmcoder.eval.metrics.{score.split(".")[0]}'), score.split(".")[1]) for score in self.config.scores if score.split(".")[0] == 'extrinsic']

        self.analysis_results: dict[str, dict] = {}

        pbar = tqdm(results.items(), desc='Analysis', total=len(results), disable=not verbose)

        for results_id, result in pbar:
            self.analysis_results[results_id] = {}
            for f in extrinsic_score_functions:
                pbar.set_description(f'Analysis: {f.__name__}')
                self.analysis_results[results_id][f.__name__] = f(ground_truth=self.targets[int(results_id)], llmcoder_result=result)  # FIXME: Support arbitrary keys instead of only integers
            for f in intrinsic_score_functions:
                pbar.set_description(f'Analysis: {f.__name__}')
                self.analysis_results[results_id][f.__name__] = f(llmcoder_result=result)

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
        for input_id, input in tqdm(inputs.items(), desc='Prediction', total=len(inputs), disable=not verbose):
            # Initialize the LLMCoder with the configuration
            # Get the completion and calture the output
            f = io.StringIO()
            with redirect_stdout(f):
                time_start = time.time()

                llmcoder = LLMCoder(
                    analyzers=self.config.get('analyzers'),
                    model_first=self.config.get('model_first'),
                    model_feedback=self.config.get('model_feedback'),
                    feedback_variant=self.config.get('feedback_variant'),
                    system_prompt=self.config.get('system_prompt'),
                    max_iter=self.config.get('max_iter'),
                    log_conversation=self.config.get('log_conversation'),
                    verbose=True
                )

                _ = llmcoder.complete(input)

                time_end = time.time()

            # Add the results to the results list
            results[input_id] = {}
            results[input_id]['messages'] = llmcoder.messages
            results[input_id]['analyzer_results'] = llmcoder.analyzer_results_history
            results[input_id]['log'] = f.getvalue()
            results[input_id]['time'] = time_end - time_start

        return results

    def _write_results(self, results: dict) -> None:
        """
        Write the results back to the database.

        Parameters
        ----------
        results : dict
            The results to write back to the database.
        """
        config_name = os.path.split(self.config.settings_file_for_dynaconf[0])[-1].split('.')[0]

        # Store the conversation in an easily parsable format
        dump_results_to_json(results, os.path.join(get_data_dir(self.config.get('dataset'), create=True), f'results_{config_name}_{self.time}', 'messages.json'))

        # Store the conversation in a human readable format
        dump_results_to_readable(results, os.path.join(get_data_dir(self.config.get('dataset'), create=True), f'results_{config_name}_{self.time}', 'readable_logs'))

    def _read_results(self, results_file: str) -> dict:
        """
        Read the results from the database.

        Parameters
        ----------
        results : dict
            The results to read from the database.
        """
        return read_results_from_json(os.path.join(get_data_dir(self.config.get('dataset')), results_file))

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
        df.to_csv(os.path.join(get_data_dir(self.config.get('dataset'), create=True), f'results_{config_name}_{self.time}', 'metrics.csv'))
