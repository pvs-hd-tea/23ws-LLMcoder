import json
import os
import numpy as np
from datetime import datetime

import openai
import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
                 log_conversation: bool = True,
                 device: str | torch.device | None = None):
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
        device : str | torch.device | None, optional
            The device to use for the scoring model, by default the first available GPU or CPU
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
        self.analyzer_pass_history: list[dict[str, dict]] = []
        self.analyzer_message_history: list[dict[str, dict]] = []
        self.max_iter = max_iter
        self.messages: list = []

        # Set up the analyzers
        if analyzers is None:
            self.analyzers = {}
        else:
            self.analyzers = {
                analyzer: AnalyzerFactory.create_analyzer(analyzer) for analyzer in analyzers
            }

        # Set up the scoring model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)

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
            self.scoring_prompt = get_system_prompt("2023-12-09_Scorer_v2.txt")
        elif scoring_prompt in os.listdir(get_system_prompt_dir()):
            self.scoring_prompt = get_system_prompt(scoring_prompt)
        else:
            self.scoring_prompt = scoring_prompt

        # Add the system prompt to the messages
        self._add_message("system", message=self.system_prompt)

    # def to(self, device: str | torch.device) -> "LLMCoder":
    #     """
    #     Move the scoring model to the specified device

    #     Parameters
    #     ----------
    #     device : str | torch.device
    #         The device to use
    #     """
    #     if isinstance(device, str):
    #         device = torch.device(device)
    #     elif isinstance(device, torch.device):
    #         self.device = device
    #     else:
    #         raise TypeError("Invalid device type")
    #     self.completion_score_model = self.completion_score_model.to(self.device)

    #     return self

    def complete(self, code: str, temperature: float = 0.7, n: int = 14) -> str:
        """
        Complete the provided code with the LLMCoder feedback loop

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 14

        Returns
        -------
        str
            The completed code
        """
        # Get the first completion with
        print("[LLMcoder] Creating first completion...")
        self.complete_first(code, temperature, n)

        print("[LLMcoder] Starting feedback loop...")
        if len(self.analyzers) > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for i in range(self.max_iter):
                print(f"[LLMcoder] Starting feedback iteration {i + 1}...")
                if self.feedback_step(temperature, n):
                    # If the feedback is correct, break the loop and return the code
                    break

        # Return the last message
        return self.messages[-1]["content"]

    @staticmethod
    def _create_conversation_file() -> str:
        """
        Create the conversation file

        Returns
        -------
        str
            The path of the conversation file
        """
        return os.path.join(get_conversations_dir(create=True), f"{datetime.now()}.jsonl")

    def score_code(self, code: str, reduction: str = "geo") -> float:
        messages = [
            {
                "role": "system",
                "content": self.scoring_prompt
            }, {
                "role": "user",
                "content": code
            }
        ]
        completions = self.client.chat.completions.create(messages=messages, model="gpt-3.5-turbo", temperature=0)

        scores = np.array([float(s) for s in completions.choices[0].message.content.split("\n")])

        match reduction:
            case "mean":
                return scores.mean()
            case "max":
                return scores.max()
            case "geo":
                return scores.prod() ** (1 / len(scores))
            case _:
                raise ValueError("Invalid reduction method")

    def _add_message(self, role: str, model: str = 'gpt-3.5-turbo', message: str | None = None, temperature: float = 0.7, n: int = 14) -> None:
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
            The number of choices to generate, by default 14
        """
        # If the user is the assistant, generate a response
        if role == "assistant" and message is None:
            chat_completions = self.client.chat.completions.create(messages=self.messages, model=model, temperature=temperature, n=n)  # type: ignore

            if n > 1:
                user_code = self.messages[1]["content"]

                # Filter out messages that already appear in the conversation
                # These are completions that have lead to an error, and we do not want to consider them again
                valid_choices = [choice for choice in chat_completions.choices if choice not in [message["content"] for message in self.messages if message["role"] == "assistant"]]

                print(f"[Scoring] Considering {len(valid_choices)} completions, ignoring {len(chat_completions.choices) - len(valid_choices)}")

                completed_code_candidates = [user_code + choice.message.content for choice in valid_choices]

                # Score the completions
                scores = np.array([self.score_code(code) for code in completed_code_candidates])

                best_message_id = scores.argmax()

                # Get the best message according to the scoring model
                message = valid_choices[best_message_id].message.content

                print(f"[Scoring] Choosing message {best_message_id} with score {scores.max()}")
            else:
                message = chat_completions.choices[0].message.content

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

    def _reset_loop(self) -> None:
        """
        Reset the feedback loop
        """
        self.iterations = 0
        self.analyzer_pass_history = []
        self.analyzer_message_history = []
        self.messages = []

        # Add the system prompt to the messages
        self._add_message("system", message=self.system_prompt)

    def _update_analyzer_history(self, analyzer_results: dict[str, dict]) -> None:
        """
        Add the analyzer results to the analyzer results list

        Parameters
        ----------
        analyzer_results : dict[dict]
            The analyzer results to add
        """
        self.iterations += 1
        self.analyzer_pass_history.append({analyzer_name: results['pass'] for analyzer_name, results in analyzer_results.items()})
        self.analyzer_message_history.append({analyzer_name: results['message'] for analyzer_name, results in analyzer_results.items()})

    def complete_first(self, code: str, temperature: float = 0.7, n: int = 14) -> str:
        """
        Run the first completion of the LLMCoder without any feedback

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 14

        Returns
        -------
        str
            The completed code
        """
        self._reset_loop()

        # We specify the user code for completion with model by default
        self._add_message("user", message=code)

        # First completion: do it changing the output format, i.e. using the fine-tuned model
        self._add_message("assistant", model=self.model_first, temperature=temperature, n=n)

        # Return the last message (the completion)
        return self.messages[-1]["content"]

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

    def feedback_step(self, temperature: float = 0.7, n: int = 14) -> bool:
        """
        Run the feedback step of the LLMCoder feedback loop

        Parameters
        ----------
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 14

        Returns
        -------
        bool
            True if the completed code passes all the analyzers, False otherwise
        """
        # Run the analyzers
        analyzer_results: dict[str, dict] = {}

        if self.feedback_variant == "separate":
            print("[LLMcoder] Analyzing code in a separate mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                print(f"[LLMcoder] Running {analyzer_name}...")
                # analyzer_results.append(analyzer.analyze(self.messages[1]['content'], self.messages[-1]['content']))
                analyzer_results[analyzer_name] = analyzer_instance.analyze(self.messages[1]['content'], self.messages[-1]['content'])
        if self.feedback_variant == "coworker":
            print("[LLMcoder] Analyzing code in a coworker mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                print(f"[LLMcoder] Running {analyzer_name}...")
                analyzer_results[analyzer_name] = analyzer_instance.analyze(self.messages[1]['content'], self.messages[-1]['content'], context=analyzer_results)

        # Print how many analyzers have passed
        print(f"[LLMcoder] {sum([results['pass'] for results in analyzer_results.values() if type(results['pass']) is bool])} / {len([results for results in analyzer_results.values() if type(results['pass']) is bool])} analyzers passed")

        # Add the analyzer results to the analyzer results list
        self._update_analyzer_history(analyzer_results)

        # Check if all the analyzers passed
        if all([results['pass'] for results in analyzer_results.values() if type(results["pass"]) is bool]):  # Could also be "ignore" or "info" for example
            # If all the analyzers passed, return True
            return True

        error_prompt = self._feedback_pattern([result['message'] for result in analyzer_results.values() if not result['pass'] or result['pass'] == "info"])

        self._add_message("user", message=error_prompt + self.messages[1]['content'])
        self._add_message("assistant", model=self.model_first, temperature=temperature, n=n)  # model_first works quite good here

        return False
