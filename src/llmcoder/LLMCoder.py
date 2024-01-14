import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import openai

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.utils import get_conversations_dir, get_openai_key, get_system_prompt, get_system_prompt_dir


class LLMCoder:
    """
    Initialize the LLMCoder

    Parameters
    ----------
    analyzers : list[str], optional
        The list of analyzers to use, by default []
    model_first : str, optional
        The model to use for the first completion, by default "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d"
    model_feedback : str, optional
        The model to use for the feedback loop, by default "gpt-3.5-turbo"
    feedback_variant : str, optional
        The feedback variant to use, by default "separate"
    system_prompt : str, optional
        The system prompt to use, by default the one used for preprocessing and fine-tuning
    max_iter : int, optional
        The maximum number of iterations to run the feedback loop, by default 10
    log_conversation : bool, optional
        Whether to log the conversation, by default False
    n_procs : int, optional
        The number of processes to use for the analyzers, by default 1
    verbose : bool, optional
        Whether to print verbose output, by default True
    """
    def __init__(self,
                 analyzers: list[str] = None,
                 model_first: str = "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d",
                 model_feedback: str = "gpt-3.5-turbo",
                 feedback_variant: str = "separate",
                 system_prompt: str | None = None,
                 max_iter: int = 10,
                 log_conversation: bool = True,
                 n_procs: int = 1,
                 verbose: bool = True) -> None:

        # Check for invalid feedback variants
        if feedback_variant not in ["separate", "coworker"]:
            raise ValueError("Inavlid feedback method")
        self.feedback_variant = feedback_variant

        # Remember the models
        self.model_first = model_first
        self.model_feedback = model_feedback

        # Set the feedback loop variables
        self.iterations = 0
        self.analyzer_results_history: list[dict[str, dict[str, float | int | str | bool]]] = []
        self.max_iter = max_iter
        self.messages: list = []

        # Set up the analyzers
        if analyzers is None:
            self.analyzers = {}
        else:
            self.analyzers = {
                analyzer: AnalyzerFactory.create_analyzer(analyzer, verbose=verbose) for analyzer in analyzers
            }

        # Set up the OpenAI API
        self.client = openai.OpenAI(api_key=get_openai_key())

        # Set up the number of processes to use for the analyzers
        self.n_procs = n_procs

        # Set up logging
        if log_conversation:
            self.conversation_file = self._create_conversation_file()
        else:
            self.conversation_file = None  # type: ignore

        # Get the system prompt
        if system_prompt is None:
            self.system_prompt = get_system_prompt()
        elif system_prompt in os.listdir(get_system_prompt_dir()):
            self.system_prompt = get_system_prompt(system_prompt)
        else:
            self.system_prompt = system_prompt

        self.verbose = verbose

        self._add_message("system", message=self.system_prompt)

    def setup(self, code: str) -> None:
        """
        Run the setup for the analyzers

        Parameters
        ----------
        code : str
            The code to analyze
        """
        for analyzer in self.analyzers.values():
            analyzer.setup(input=code)

    def _check_passing(self) -> bool:
        """
        Check if all the analyzers passed in the last iteration

        Returns
        -------
        bool
            True if all the analyzers passed, False otherwise
        """
        # If there was no iteration yet, return True
        if len(self.analyzer_results_history) == 0:
            return True

        # Print how many analyzers have passed
        n_passed = sum(results['pass'] for results in self.analyzer_results_history[-1].values()
                       if (results['type'] == "critical" and type(results['pass']) is bool))
        n_total = len([results for results in self.analyzer_results_history[-1].values()
                      if results['type'] == "critical"])

        if self.verbose:
            print(f"[LLMcoder] {n_passed} / {n_total} analyzers passed")

        # If all the analyzers passed, return True
        if n_passed == n_total:
            return True

        # Otherwise, return False
        return False

    def complete(self, code: str, temperature: float = 0.7, n: int = 1) -> str:
        """
        Complete the provided code with the LLMCoder feedback loop

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 1

        Returns
        -------
        str
            The completed code
        """
        # Reset the feedback loop and internal variables
        self._reset_loop()

        # Get the first completion with
        if self.verbose:
            print("[LLMcoder] Creating first completion...")
        completion = self.step(code, temperature, n)

        if completion is None:
            raise RuntimeError("Completion generation failed")

        # If the first completion is already correct, return it
        if self._check_passing():
            return self.messages[-1]["content"]

        # Otherwise, start the feedback loop (but only if there are analyzers that can be used)
        if self.verbose:
            print("[LLMcoder] Starting feedback loop...")
        if len(self.analyzers) > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for i in range(self.max_iter):
                if self.verbose:
                    print(f"[LLMcoder] Starting feedback iteration {i + 1}...")

                completion = self.step(code, temperature, n)

                if completion is None:
                    if self.verbose:
                        print("[LLMcoder] Completion generation failed. Stopping early...")
                    break

                # If the code is correct, break the loop
                if self._check_passing():
                    break

        # Return the last message regardless of whether it is correct or not
        return self.messages[-1]["content"]

    @classmethod
    def _create_conversation_file(cls) -> str:
        """
        Create the conversation file

        Returns
        -------
        str
            The path to the conversation file
        """
        return os.path.join(get_conversations_dir(create=True), f"{datetime.now()}.jsonl")

    def _is_bad_completion(self, completion: str) -> bool:
        """
        Check if the completion already appeared in the conversation. If the assistant repeats a mistake, we do not want to consider it again

        Parameters
        ----------
        completion : str
            The completion to check

        Returns
        -------
        bool
            True if the completion already appeared in the conversation, False otherwise
        """
        return completion in [message["content"] for message in self.messages if message["role"] == "assistant"]

    def _get_completions(self, model: str = 'gpt-3.5-turbo', temperature: float = 0.7, n: int = 1) -> str | None:
        """
        Use OpenAI's API to get completion(s) for the user's code

        Parameters
        ----------
        model : str, optional
            The model to use for the completion, by default 'gpt-3.5-turbo'
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 1

        Returns
        -------
        str | None
            The completion(s) or None if all completions are repetitions of previous mistakes
        """
        # Get the completions from OpenAI's API
        candidates = self.client.chat.completions.create(messages=self.messages, model=model, temperature=temperature, n=n)  # type: ignore

        # Filter out completions that are repetitions of previous mistakes
        valid_choices = [completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]

        # If all completions are repetitions of previous mistakes, increase the temperature and the number of choices until we get a valid completion
        increased_temperature = temperature
        increased_n = n
        MAX_RETRIES = 10
        repetition = 0

        while len(valid_choices) == 0 and repetition < MAX_RETRIES:
            if self.verbose:
                print(f"[LLMcoder] All completions are repetitions of previous mistakes. Increasing temperature to {increased_temperature} and number of choices to {increased_n}... [repetition {repetition + 1}/{MAX_RETRIES}]")
            candidates = self.client.chat.completions.create(messages=self.messages, model=model, temperature=increased_temperature, n=increased_n)  # type: ignore
            valid_choices = [completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]

            increased_temperature = min(2, increased_temperature + 0.1)
            increased_n = min(32, increased_n * 2)
            repetition += 1

        # If we still do not have valid choices, abort
        if repetition >= MAX_RETRIES:
            if self.verbose:
                print("[LLMcoder] All completions are repetitions of previous mistakes. Aborting...")
            return None

        # Now that we have valid choices, run the analyzers on them in parallel and determine the best one
        if n > 1 and len(valid_choices) > 1:
            if self.verbose:
                print(f"[LLMcoder] Analyzing {len(candidates.choices)} completions...")

            with ThreadPoolExecutor(max_workers=self.n_procs) as executor:
                # Create a mapping of future to completion choice
                choice_to_future = {
                    i: executor.submit(self.run_analyzers, self.messages[1]["content"], choice.message.content)
                    for i, choice in enumerate(valid_choices)
                }

                # Retrieve results in the order of valid_choices
                analysis_results_list = []
                for i in range(len(valid_choices)):
                    future = choice_to_future[i]
                    try:
                        analysis_results = future.result()
                        analysis_results_list.append(analysis_results)
                    except Exception as exc:
                        if self.verbose:
                            print(f"[LLMcoder] An exception occurred during analysis of choice: {exc}")
                        raise exc
                        # analysis_results_list.append({
                        #     "score": -np.inf,
                        #     "type": "ignore",
                        #     "message": "An exception occured during analysis",
                        #     "pass": False,
                        # })

            # Choose the completion with the highest score
            candidate_scores = [sum([results["score"] for results in result.values()]) for result in analysis_results_list]
            best_completion_id = np.argmax(candidate_scores)
            if self.verbose:
                print(f"[Scoring] Choosing message {best_completion_id} with score {candidate_scores[best_completion_id]}")

            # Select the best completion
            message = valid_choices[best_completion_id].message.content

            # Update the analyzer results history with the results of the best completion
            self.analyzer_results_history.append(analysis_results_list[best_completion_id])
        else:
            # If we only have one completion, still run the analyzers on it
            if self.verbose:
                print("[LLMcoder] Analyzing completion...")
            analysis_results = self.run_analyzers(self.messages[1]["content"], valid_choices[0].message.content)

            # There is only one valid choice
            message = valid_choices[0].message.content

            # Update the analyzer results history with the results of the completion
            self.analyzer_results_history.append(analysis_results)

        # Return the best (or only) completion
        return message

    def _add_message(self, role: str, message: str | None = None, model: str = 'gpt-3.5-turbo', temperature: float = 0.7, n: int = 1) -> bool:
        """
        Add a message to the messages list. Used as a unified way to add messages to the conversation

        Parameters
        ----------
        role : str
            The role of the message
        message : str, optional
            The message to add, by default None
        model : str, optional
            The model to use for the completion, by default 'gpt-3.5-turbo'
        temperature : float, optional
            The temperature to use for the assistant completion, by default 0.7
        n : int, optional
            The number of assistant choices to generate, by default 1

        Returns
        -------
        bool
            True if the message was added, False otherwise
        """
        # If the user is the assistant, generate a response
        if role == "assistant" and message is None:
            message = self._get_completions(model, temperature, n)

            # If the completion generation did not work, abort
            if message is None:
                return False

        # If the role is user or system, or if the assistant message should be overwritten, add the message
        self.messages.append(
            {
                "role": role,
                "content": message,
            }
        )

        if self.verbose:
            print(f"[LLMcoder] {role.upper()}: {message}")

        # If the conversation should be logged, log it
        if self.conversation_file is not None:
            # If the conversation file already exists, only append the last message as a single line
            if os.path.isfile(self.conversation_file):
                with open(self.conversation_file, "a") as f:
                    f.write(json.dumps(self.messages[-1], ensure_ascii=False) + "\n")
            # Otherwise, write the whole conversation
            else:
                with open(self.conversation_file, "w") as f:
                    for message in self.messages:
                        f.write(json.dumps(message, ensure_ascii=False) + "\n")

        # The message was added successfully
        return True

    def _reset_loop(self) -> None:
        """
        Reset the feedback loop and internal variables. Also re-add the system prompt as the first message
        """
        # Reset the feedback loop variables
        self.iterations = 0
        self.analyzer_results_history = []
        self.messages = []

        # Add the system prompt to the messages
        self._add_message("system", message=self.system_prompt)

    def run_analyzers(self, code: str, completion: str) -> dict[str, dict]:
        """
        Run the analyzers on the code and completion

        Parameters
        ----------
        code : str
            The code to analyze
        completion : str
            The completion to analyze

        Returns
        -------
        dict[str, dict]
            The analyzer results
        """
        analyzer_results: dict[str, dict] = {}

        # In separete mode, each analyzer is run separately without a shared context
        if self.feedback_variant == "separate":
            if self.verbose:
                print("[LLMcoder] Analyzing code in separate mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                if self.verbose:
                    print(f"[LLMcoder] Running {analyzer_name}...")
                analyzer_results[analyzer_name] = analyzer_instance.analyze(code, completion)

        # In coworker mode, the analyzers are run in parallel and share a context
        elif self.feedback_variant == "coworker":
            if self.verbose:
                print("[LLMcoder] Analyzing code in coworker mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                if self.verbose:
                    print(f"[LLMcoder] Running {analyzer_name}...")
                analyzer_results[analyzer_name] = analyzer_instance.analyze(code, completion, context=analyzer_results)

        # Return the collected analyzer results
        return analyzer_results

    def _feedback_prompt_template(self, result_messages: list[str]) -> str:
        """
        Create the feedback prompt template for the analyzer results

        Parameters
        ----------
        result_messages : list[str]
            The analyzer result messages

        Returns
        -------
        str
            The feedback prompt template
        """
        return '[INST]\n' + '\n'.join(result_messages) + '\n\nFix, improve and rewrite your completion for the following code:\n[/INST]\n'

    def step(self, code: str, temperature: float = 0.7, n: int = 1) -> str | None:
        """
        Complete the provided code with the OpenAI model and feedback, if available

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 1

        Returns
        -------
        str | None
            The completed code or None if the completion generation failed
        """
        # If there is feedback available from previous analyses, add it to the prompt
        if len(self.analyzer_results_history) > 0:
            feedback_prompt = self._feedback_prompt_template([str(result['message']) for result in self.analyzer_results_history[-1].values()
                                                              if (not result['pass'] or result['type'] == "info")])

        # If there is not feedback available, the prompt will just be the user's code
        else:
            feedback_prompt = ""

        # Add the prompt to the messages
        self._add_message("user", message=feedback_prompt + code)

        # Get a completion from the assistant
        success = self._add_message("assistant", model=self.model_first, temperature=temperature, n=n)  # model_first works quite good here

        self.iterations += 1

        # If the completion generation failed, abort
        if not success:
            return None

        # Return the last message (the completion)
        return self.messages[-1]["content"]
