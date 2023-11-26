from dynaconf import Dynaconf

from llmcoder import LLMCoder  # Import your LLMCoder class


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

    def run(self) -> None:
        """
        Run the evaluation end to end (reading inputs from the database and writing results back)
        """
        # inputs = self._get_inputs()
        # results = self._run(inputs)
        # self._write_results(results)
        raise NotImplementedError

    def analyze(self) -> None:
        """
        Analyze the results from the database given the configuration and store it back in the database.
        """
        # inputs = self._get_inputs()
        # results = self._get_results()

        # Count tokens
        # Count the number of iterations
        # Determine the quality of the completion
        # - Number of comments
        # - Number of lines
        # - Number of tokens
        # - Readability
        # Compare the completion to the ground truth
        # - Levenshtein distance
        # - BLEU score
        # - Semantic similarity via BERT or similar encoder

        raise NotImplementedError

    def _get_results(self) -> list[dict]:
        """
        Get the results from the database.

        Returns
        -------
        List[dict]
            A list of results from the database.
        """
        raise NotImplementedError

    def _get_inputs(self) -> list[str]:
        """
        Get the inputs to run the LLMCoder on.

        Returns
        -------
        List[str]
            A list of file paths to run the LLMCoder on.
        """
        raise NotImplementedError

    def _run(self, inputs: dict) -> dict:
        """
        Run the LLMCoder on the provided files and return the results.

        Parameters
        ----------
        inputs : List[str]
            A list of inputs to complete with the LLMCoder.
        """

        results: dict[str, dict] = {}

        # Run the LLMCoder on each input
        for input_id, input in inputs.items():
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

    def _write_results(self, results: list[dict]) -> None:
        """
        Write the results back to the database.

        Parameters
        ----------
        results : List[str]
            A list of results to write back to the database.
        """
        raise NotImplementedError
