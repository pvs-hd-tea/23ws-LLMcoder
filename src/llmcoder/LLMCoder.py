import json
import os
import numpy as np
import openai

from datetime import datetime

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.utils import get_conversations_dir, get_openai_key, get_system_prompt, get_system_prompt_dir


class LLMCoder:
    def __init__(self,
                 analyzers: list[str] = None,
                 model_first: str = "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d",
                 model_feedback: str = "gpt-3.5-turbo",
                 feedback_variant: str = "separate",
                 system_prompt: str | None = None,
                 scoring_prompt: str | None = None,
                 max_iter: int = 10,
                 log_conversation: bool = True):
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
        scoring_prompt : str, optional
            The scoring prompt to use, by default the one used for scoring
        max_iter : int, optional
            The maximum number of iterations to run the feedback loop, by default 10
        log_conversation : bool, optional
            Whether to log the conversation, by default False
        """
        # Check for invalid feedback variants
        if feedback_variant not in ["separate", "coworker"]:
            raise ValueError("Inavlid feedback method")
        self.feedback_variant = feedback_variant

        # Remember the models
        self.model_first = model_first
        self.model_feedback = model_feedback

        # Set the feedback loop variables
        self.iterations = 0
        self.analyzer_results_history: list[list[dict[str, float | int | str | bool]]] = []
        self.max_iter = max_iter
        self.messages: list = []

        # Set up the analyzers
        if analyzers is None:
            self.analyzers = {}
        else:
            self.analyzers = {
                analyzer: AnalyzerFactory.create_analyzer(analyzer) for analyzer in analyzers
            }

        # Set up the OpenAI API
        self.client = openai.OpenAI(api_key=get_openai_key())

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

        if scoring_prompt is None:
            self.scoring_prompt = get_system_prompt("2023-12-09_Scorer_v1.1.txt")
        elif scoring_prompt in os.listdir(get_system_prompt_dir()):
            self.scoring_prompt = get_system_prompt(scoring_prompt)
        else:
            self.scoring_prompt = scoring_prompt

    def _check_passing(self) -> bool:
        if len(self.analyzer_results_history) == 0:
            return False

        # Print how many analyzers have passed
        n_passed = sum([results['pass'] for results in self.analyzer_results_history[-1].values() if results['type'] == "critical"])
        n_total = len([results for results in self.analyzer_results_history[-1].values() if results['type'] == "critical"])
        print(f"[LLMcoder] {n_passed} / {n_total} analyzers passed")

        # Check if all the analyzers passed
        if n_passed == n_total:
            # If all the analyzers passed, return True
            return True

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
        # Reset the feedback loop
        self._reset_loop()

        # Get the first completion with
        print("[LLMcoder] Creating first completion...")
        self.step(code, temperature, n)

        if self._check_passing():
            return self.messages[-1]["content"]

        print("[LLMcoder] Starting feedback loop...")
        if len(self.analyzers) > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for i in range(self.max_iter):
                print(f"[LLMcoder] Starting feedback iteration {i + 1}...")

                self.step(code, temperature, n)

                if self._check_passing():
                    break

        # Return the last message
        return self.messages[-1]["content"]

    @classmethod
    def _create_conversation_file(cls) -> str:
        """
        Create the conversation file

        Returns
        -------
        str
            The path of the conversation file
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
        candidates = self.client.chat.completions.create(messages=self.messages, model=model, temperature=temperature, n=n)  # type: ignore
        valid_choices = [completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]

        # If all completions are repetitions of previous mistakes, increase the temperature and the number of choices until we get a valid completion
        increased_temperature = temperature
        increased_n = n
        MAX_RETRIES = 10
        repetition = 0

        while len(valid_choices) == 0 and repetition < MAX_RETRIES:
            print(f"[LLMcoder] All completions are repetitions of previous mistakes. Increasing temperature to {increased_temperature} and number of choices to {increased_n}... [repetition {repetition + 1}/{MAX_RETRIES}]")
            candidates = self.client.chat.completions.create(messages=self.messages, model=model, temperature=increased_temperature, n=increased_n)  # type: ignore
            valid_choices = [completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]

            increased_temperature = min(2, increased_temperature + 0.1)
            increased_n = min(32, increased_n * 2)
            repetition += 1

        if repetition >= MAX_RETRIES:
            print("[LLMcoder] All completions are repetitions of previous mistakes. Aborting...")
            return None

        # Now that we have valid choices, run the analyzers on them in parallel and determine the best one
        if n > 1:
            analysis_results = []
            print(f"[LLMcoder] Analyzing {len(candidates.choices)} completions...")

            for i, choice in enumerate(valid_choices):
                print(f"[LLMcoder] Analyzing completion {i}...")
                analysis_results.append(self.run_analyzers(self.messages[1]["content"], choice.message.content))

            # Choose the completion with the highest score
            candidate_scores = [sum([results["score"] for results in result.values()]) for result in analysis_results]
            best_completion_id = np.argmax(candidate_scores)

            message = valid_choices[best_completion_id].message.content

            self.analyzer_results_history.append(analysis_results[best_completion_id])

            print(f"[Scoring] Choosing message {best_completion_id} with score {candidate_scores[best_completion_id]}")
        else:
            print("[LLMcoder] Analyzing completion...")
            analysis_results = self.run_analyzers(self.messages[1]["content"], valid_choices[0].message.content)

            self.analyzer_results_history.append(analysis_results)

            message = valid_choices[0].message.content

        return message

    def _add_message(self, role: str, model: str = 'gpt-3.5-turbo', message: str | None = None, temperature: float = 0.7, n: int = 1) -> bool:
        """
        Add a message to the messages list

        Parameters
        ----------
        role : str
            The role of the message
        model : str, optional
            The model to use for the completion, by default 'gpt-3.5-turbo'
        message : str, optional
            The message to add, by default None
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 1

        Returns
        -------
        bool
            True if the message was added, False otherwise
        """
        # If the user is the assistant, generate a response
        if role == "assistant" and message is None:
            message = self._get_completions(model, temperature, n)

        self.messages.append(
            {
                "role": role,
                "content": message,
            }
        )

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

        return True

    def _reset_loop(self) -> None:
        """
        Reset the feedback loop
        """
        self.iterations = 0
        self.analyzer_results_history = []
        self.messages = []

        # Add the system prompt to the messages
        self._add_message("system", message=self.system_prompt)

    def run_analyzers(self, code: str, completion: str) -> dict[str, dict]:
        analyzer_results: dict[str, dict] = {}

        if self.feedback_variant == "separate":
            print("[LLMcoder] Analyzing code in separate mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                print(f"[LLMcoder] Running {analyzer_name}...")
                analyzer_results[analyzer_name] = analyzer_instance.analyze(code, completion)
        if self.feedback_variant == "coworker":
            print("[LLMcoder] Analyzing code in coworker mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                print(f"[LLMcoder] Running {analyzer_name}...")
                analyzer_results[analyzer_name] = analyzer_instance.analyze(code, completion, context=analyzer_results)

        return analyzer_results

    def _feedback_pattern(self, result_messages: list[str]) -> str:
        """
        Create the feedback pattern for the analyzer results

        Parameters
        ----------
        result_messages : list[str]
            The analyzer result messages

        Returns
        -------
        str
            The feedback pattern
        """
        return '[INST]\n' + '\n'.join(result_messages) + '\n\nFix, improve and rewrite your completion for the following code:\n[/INST]\n'

    def step(self, code: str, temperature: float = 0.7, n: int = 1) -> str:
        """
        Run the first completion of the LLMCoder without any feedback

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
        if len(self.analyzer_results_history) > 0:
            feedback_prompt = self._feedback_pattern([result['message'] for result in self.analyzer_results_history[-1].values() if not result['pass'] or result['type'] == "info"])
        else:
            feedback_prompt = ""

        self._add_message("user", message=feedback_prompt + code)
        self._add_message("assistant", model=self.model_first, temperature=temperature, n=n)  # model_first works quite good here

        # Return the last message (the completion)
        return self.messages[-1]["content"]
