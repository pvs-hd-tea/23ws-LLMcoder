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
from llmcoder.llmcoder import LLMCoder  # Import your LLMCoder class
from llmcoder.utils import get_config_dir, get_data_dir


def check_config(config: Dynaconf) -> bool:

    # Return if all the required keys are present and the types are correct
    if not isinstance(config.get('analyzers'), list):
        raise TypeError(f'Error when checking config: analyzers should be a list, but got {type(config.get("analyzers"))}')
    if not isinstance(config.get('model_first'), str):
        raise TypeError(f'Error when checking config: model_first should be a string, but got {type(config.get("model_first"))}')
    if not isinstance(config.get('model_feedback'), str):
        raise TypeError(f'Error when checking config: model_feedback should be a string, but got {type(config.get("model_feedback"))}')
    if not isinstance(config.get('feedback_variant'), str):
        raise TypeError(f'Error when checking config: feedback_variant should be a string, but got {type(config.get("feedback_variant"))}')
    if not isinstance(config.get('system_prompt'), str):
        raise TypeError(f'Error when checking config: system_prompt should be a string, but got {type(config.get("system_prompt"))}')
    if not isinstance(config.get('dataset'), str):
        raise TypeError(f'Error when checking config: dataset should be a string, but got {type(config.get("dataset"))}')
    if not isinstance(config.get('max_iter'), int):
        raise TypeError(f'Error when checking config: max_iter should be an int, but got {type(config.get("max_iter"))}')
    if not isinstance(config.get('log_conversation'), bool):
        raise TypeError(f'Error when checking config: log_conversation should be a bool, but got {type(config.get("log_conversation"))}')
    if not isinstance(config.get('scores'), list):
        raise TypeError(f'Error when checking config: scores should be a list, but got {type(config.get("scores"))}')
    if not isinstance(config.get('n_choices'), int):
        raise TypeError(f'Error when checking config: n_choices should be an int, but got {type(config.get("n_choices"))}')
    if not isinstance(config.get('n_procs'), int):
        raise TypeError(f'Error when checking config: n_procs should be an int, but got {type(config.get("n_procs"))}')

    return True


class Evaluation:
    def __init__(self, configs: Dynaconf | list[Dynaconf] | None = None):
        """
        Initialize the Evaluation with a Dynaconf configuration.

        Parameters
        ----------
        configs : Dynaconf | list[Dynaconf], optional
            The configuration object from Dynaconf.
        """
        if configs is None:
            # Load all configurations from the config directory
            self.configs = [
                Dynaconf(settings_files=[os.path.join(get_config_dir(), config)])
                for config in sorted(os.listdir(get_config_dir())) if config.endswith('.yaml')]
        elif isinstance(configs, Dynaconf):
            self.configs = [configs]
        elif isinstance(configs, str):
            # Check if the config file exists
            if not os.path.exists(os.path.join(get_config_dir(), configs)):
                raise FileNotFoundError(f'Config file not found at {os.path.join(get_config_dir(), configs)}')
            self.configs = [Dynaconf(settings_files=[os.path.join(get_config_dir(), configs)])]
        else:
            self.configs = configs

        # Check if the configuration is correct.
        for config in self.configs:
            check_config(config)

        # Check if the datasets exists
        for config in self.configs:
            dataset_path = os.path.abspath(os.path.join(get_data_dir(config.get("dataset")), 'conversations.jsonl'))
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f'Dataset not found at {dataset_path}')

        print('Set up evaluation with configurations:')
        for config in self.configs:
            print(f'\t- {config.settings_file_for_dynaconf[0]}')

    def run(self, store: bool = True, n_repeat: int = 1, verbose: bool = False) -> dict[str, list]:
        """
        Run the evaluation end to end (reading inputs from the database and writing results back)

        Parameters
        ----------
        store : bool, optional
            Whether to store the results in the database, by default True
        n_repeat : int, optional
            The number of times to repeat the evaluation, by default 1
        verbose : bool, optional
            Whether to print the results to the console, by default False

        Returns
        -------
        dict[str, list]
            The results from the evaluation.
        """
        results: dict[str, list] = {}
        # Run the evaluation n_repeat times
        for i in range(n_repeat):
            print(f'Running evaluation {i+1}/{n_repeat}')
            for config in self.configs:
                print(f'Running evaluation for configuration: {config.settings_file_for_dynaconf[0]}')
                # Run the LLMCoder on the user inputs
                result = self.predict(config=config, store=store, verbose=verbose)

                time.sleep(2)

                if config.settings_file_for_dynaconf[0] not in results:
                    results[config.settings_file_for_dynaconf[0]] = []

                results[config.settings_file_for_dynaconf[0]].append(result)

        # Analyze the results
        return results

    def predict(self, config: Dynaconf, store: bool = False, verbose: bool = False) -> dict:
        """
        Run the LLMCoder on the provided files and write the results to the database.

        Parameters
        ----------
        config : Dynaconf
            The configuration object from Dynaconf.
        store : bool, optional
            Whether to store the results in the database, by default False
        verbose : bool, optional
            Whether to print the results to the console, by default False

        Returns
        -------
        dict
            The results from the evaluation.
        """
        # Get the data to run the LLMCoder on
        data = read_data_from_conversations_file(os.path.join(
            get_data_dir(config.get('dataset')),
            'conversations.jsonl')
        )
        inputs = {i: input for i, (input, _) in enumerate(data)}

        # Run the LLMCoder on the user inputs
        results = self.run_llmcoder(config=config, inputs=inputs, verbose=verbose)

        # Store the results in the database
        if store:
            self._write_results(config, results)

        # Return the results
        return results

    def run_llmcoder(self, config: Dynaconf, inputs: dict, verbose: bool = False) -> dict:
        """
        Run the LLMCoder on the provided files and return the results.

        Parameters
        ----------
        config : Dynaconf
            The configuration object from Dynaconf.
        inputs : List[str]
            A list of inputs to complete with the LLMCoder.
        verbose : bool, optional
            Whether to print the results to the console, by default False

        Returns
        -------
        dict
            The results from the evaluation.
        """
        results: dict[str, dict] = {}

        # Run the LLMCoder on each input
        for input_id, input in tqdm(inputs.items(), desc='Prediction', total=len(inputs), disable=not verbose):
            # Get the completion and capture the output
            f = io.StringIO()

            with redirect_stdout(f):
                time_start = time.time()

                # Initialize the LLMCoder with the configuration
                llmcoder = LLMCoder(
                    analyzers=config.get('analyzers'),
                    model_first=config.get('model_first'),
                    model_feedback=config.get('model_feedback'),
                    feedback_variant=config.get('feedback_variant'),
                    system_prompt=config.get('system_prompt'),
                    max_iter=config.get('max_iter'),
                    backtracking=config.get('backtracking'),
                    log_conversation=config.get('log_conversation'),
                    n_procs=config.get('n_procs'),
                    verbose=True
                )

                try:
                    _ = llmcoder.complete(input, n=config.get('n_choices'))
                except TypeError as e:
                    with open(os.path.join(get_data_dir(config.get('dataset'), create=True), 'error.log'), 'a') as file:
                        file.write(f'Error while running LLMCoder on input {input_id}:\n')
                        file.write(f'{e}\n')
                        file.write(f'{f.getvalue()}\n')
                        file.write(f'{"-"*80}\n')

                time_end = time.time()

            # Add the results to the results list
            results[input_id] = {}
            results[input_id]['messages'] = llmcoder.conversations.pop(keep=True).messages
            results[input_id]['analyzer_results'] = llmcoder.conversations.pop(keep=True).analyses
            results[input_id]['log'] = f.getvalue()
            results[input_id]['time'] = time_end - time_start
            results[input_id]['n_tokens_generated'] = llmcoder.n_tokens_generated

            time.sleep(1)  # Avoid API rate limits

        return results

    def _write_results(self, config: Dynaconf, results: dict) -> None:
        """
        Write the results back to the database.

        Parameters
        ----------
        results : dict
            The results to write back to the database.
        """
        # Get the current time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        config_name = os.path.splitext(os.path.split(config.settings_file_for_dynaconf[0])[-1])[0]

        # Store the conversation in an easily parsable format
        dump_results_to_json(results, os.path.join(get_data_dir(config.get('dataset'), create=True), 'eval', f'{config_name}', f'{current_time}', 'results.json'))

        # Store the conversation in a human readable format
        dump_results_to_readable(results, os.path.join(get_data_dir(config.get('dataset'), create=True), 'eval', f'{config_name}', f'{current_time}', 'readable_logs'))


class Metrics:
    def __init__(self, configs: Dynaconf | list[Dynaconf] | None = None):
        """
        Initialize the Metrics with a Dynaconf configuration.

        Parameters
        ----------
        configs : Dynaconf | list[Dynaconf], optional
            The configuration object from Dynaconf.
        """
        if configs is None:
            # Load all configurations from the config directory
            self.configs = [
                Dynaconf(settings_files=[os.path.join(get_config_dir(), config)])
                for config in sorted(os.listdir(get_config_dir())) if config.endswith('.yaml')]
        elif isinstance(configs, Dynaconf):
            self.configs = [configs]
        elif isinstance(configs, str):
            # Check if the config file exists
            if not os.path.exists(os.path.join(get_config_dir(), configs)):
                raise FileNotFoundError(f'Config file not found at {os.path.join(get_config_dir(), configs)}')
            self.configs = [Dynaconf(settings_files=[os.path.join(get_config_dir(), configs)])]
        else:
            self.configs = configs

        # Check if the configuration is correct.
        for config in self.configs:
            check_config(config)

        # Check if the datasets exists
        for config in self.configs:
            dataset_path = os.path.abspath(os.path.join(get_data_dir(config.get("dataset")), 'conversations.jsonl'))
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f'Dataset not found at {dataset_path}')

        print('Set up metrics with configurations:')
        for config in self.configs:
            print(f'\t- {config.settings_file_for_dynaconf[0]}')

    def run(self, store: bool = False, index: int | None = None, verbose: bool = False) -> dict[str, dict[str, dict[str, dict]]]:
        """
        Analyze the results from the database given the configuration and store it back in the database.

        Parameters
        ----------
        store : bool, optional
            Whether to store the analysis in the database, by default False
        index : int, optional
            The index of the results to analyze, by default None (analyze all)
        verbose : bool, optional
            Whether to print the results to the console, by default False

        Returns
        -------
        dict[str, dict[str, dict[str, dict]]]
            The analysis results.
        """

        metrics = {}
        # Run the evaluation n_repeat times for each configuration
        for config in self.configs:
            print(f'Analyzing results for configuration: {config.settings_file_for_dynaconf[0]}')

            # The data only needs to be loaded once per configuration
            data = read_data_from_conversations_file(os.path.join(
                get_data_dir(config.get('dataset')),
                'conversations.jsonl')
            )
            targets = {i: target for i, (_, target) in enumerate(data)}

            # Load the results (including multiple repititions) from the database
            results = self.load_results(config, index=index)

            metric = self.compute_metrics(config, results, targets, store=store, verbose=verbose)
            metrics[config.settings_file_for_dynaconf[0]] = metric

        return metrics

    def compute_metrics(self, config: Dynaconf, results_dict: dict[str, dict], targets: dict, store: bool = False, verbose: bool = False) -> dict[str, dict[str, dict]]:
        """
        Compute the metrics for the results.

        Parameters
        ----------
        config : Dynaconf
            The configuration object from Dynaconf.
        results_dict : dict[str, dict]
            The results to compute the metrics for.
        targets : dict
            The target completions to compute with the LLMCoder.
        intrinsic_score_functions : list[callable]
            The intrinsic score functions to use.
        extrinsic_score_functions : list[callable]
            The extrinsic score functions to use.
        store : bool, optional
            Whether to store the analysis in the database, by default False
        verbose : bool, optional
            Whether to print the results to the console, by default False

        Returns
        -------
        dict[str, dict[str, dict]]
            The metrics for each result.
        """

        metrics_dict = {}

        for result_repitition_id, results in results_dict.items():

            metrics: dict[str, dict] = {}

            intrinsic_score_functions = [getattr(importlib.import_module(f'llmcoder.eval.metrics.{score.split(".")[0]}'), score.split(".")[1]) for score in config.scores if score.split(".")[0] == 'intrinsic']
            extrinsic_score_functions = [getattr(importlib.import_module(f'llmcoder.eval.metrics.{score.split(".")[0]}'), score.split(".")[1]) for score in config.scores if score.split(".")[0] == 'extrinsic']

            pbar = tqdm(results.items(), desc='Analysis', total=len(results), disable=not verbose)

            for results_id, result in pbar:
                metrics[results_id] = {}
                for f in extrinsic_score_functions:
                    pbar.set_description(f'Analysis: {f.__name__}')
                    metrics[results_id][f.__name__] = f(ground_truth=targets[int(results_id)], llmcoder_result=result)  # FIXME: Support arbitrary keys instead of only integers
                for f in intrinsic_score_functions:
                    pbar.set_description(f'Analysis: {f.__name__}')
                    metrics[results_id][f.__name__] = f(llmcoder_result=result)

            if store:
                self._write_metrics(config, result_repitition_id, metrics)

            time.sleep(2)

            metrics_dict[result_repitition_id] = metrics

        return metrics_dict

    def load_results(self, config: Dynaconf, index: int | list[int] | None = None) -> dict[str, dict]:
        """
        Read the results for a tuple from the database.

        Parameters
        ----------
        config : Dynaconf
            The configuration object from Dynaconf.
        index : int | list[int], optional
            The index of the results to read, by default None (read all)

        Returns
        -------
        dict[str, dict]
            The results from the database.
        """
        config_name = os.path.splitext(os.path.split(config.settings_file_for_dynaconf[0])[-1])[0]

        # List the directories in the results_<dataset> folder
        results_dirs = sorted(os.listdir(os.path.join(get_data_dir(config.get('dataset')), 'eval', config_name)))

        if index is not None:
            if isinstance(index, int):
                results_dirs = [results_dirs[index]]
            elif isinstance(index, list):
                results_dirs = [results_dirs[i] for i in index]

        results = {}
        for results_dir in results_dirs:
            results[results_dir] = read_results_from_json(os.path.join(get_data_dir(config.get('dataset')), 'eval', config_name, results_dir, 'results.json'))

        return results

    def _write_metrics(self, config: Dynaconf, result_repitition_id: str, metrics: dict) -> None:
        """
        Write the analysis results back to the database.

        Parameters
        ----------
        config : Dynaconf
            The configuration object from Dynaconf.
        result_repitition_id : str
            The id of the result repitition.
        metrics : dict
            The analysis results to write back to the database.
        """
        # Create a dataframe from the analysis results
        df = pd.DataFrame.from_dict(metrics, orient='index')

        # Write the dataframe to the database
        config_name = os.path.splitext(os.path.split(config.settings_file_for_dynaconf[0])[-1])[0]
        df.to_csv(os.path.join(get_data_dir(config.get('dataset'), create=True), 'eval', f'{config_name}', f'{result_repitition_id}', 'metrics.csv'))
