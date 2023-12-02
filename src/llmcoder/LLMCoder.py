import json
import os
from datetime import datetime

import openai

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.utils import get_conversations_dir, get_openai_key, get_system_prompt, get_system_prompt_dir


class LLMCoder:
    def __init__(self,
                 analyzers: list[str] = None,
                 model_first: str = "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d",
                 model_feedback: str = "gpt-3.5-turbo",
                 feedback_variant: str = "separate",
                 system_prompt: str | None = None,
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
        max_iter : int, optional
            The maximum number of iterations to run the feedback loop, by default 10
        log_conversation : bool, optional
            Whether to log the conversation, by default False
        """
        if analyzers is None:
            self.analyzers = []
        else:
            self.analyzers = [AnalyzerFactory.create_analyzer(analyzer) for analyzer in analyzers]

        self.model_first = model_first
        self.model_feedback = model_feedback

        self.client = openai.OpenAI(api_key=get_openai_key())
        if feedback_variant not in ["separate", "coworker"]:
            raise ValueError("Inavlid feedback method")

        self.iterations = 0
        self.analyzer_pass_history: list[list[dict]] = []
        self.max_iter = max_iter
        self.feedback_variant = feedback_variant

        if log_conversation:
            self.conversation_file = self._create_conversation_file()
        else:
            self.conversation_file = None  # type: ignore

        self.messages: list = []

        if system_prompt is None:
            self.system_prompt = get_system_prompt()
        elif system_prompt in os.listdir(get_system_prompt_dir()):
            self.system_prompt = get_system_prompt(system_prompt)
        else:
            self.system_prompt = system_prompt

        self._add_message("system", message=self.system_prompt)

    def complete(self, code: str) -> str:
        """
        Complete the provided code with the LLMCoder feedback loop

        Parameters
        ----------
        code : str
            The code to complete

        Returns
        -------
        str
            The completed code
        """
        # Get the first completion with
        self.complete_first(code)

        if len(self.analyzers) > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for _ in range(self.max_iter):
                if self.feedback_step():
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

    def _add_message(self, role: str, model: str = 'gpt-3.5-turbo', message: str | None = None) -> None:
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
        """
        # If the user is the assistant, generate a response
        if role == "assistant" and message is None:
            chat_completion = self.client.chat.completions.create(messages=self.messages, model=model)  # type: ignore

            message = chat_completion.choices[0].message.content

        self.messages.append({
            "role": role,
            "content": message,
        })

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

    def _update_analyzer_pass_history(self, analyzer_results: list[dict]) -> None:
        """
        Add the analyzer results to the analyzer results list

        Parameters
        ----------
        analyzer_results : list[dict]
            The analyzer results to add
        """
        self.iterations += 1
        self.analyzer_pass_history.append([results['pass'] for results in analyzer_results])

    def complete_first(self, code: str) -> dict:
        """
        Run the first completion of the LLMCoder without any feedback

        Parameters
        ----------
        code : str
            The code to complete

        Returns
        -------
        dict
            The message of the assistant
        """
        self._reset_loop()

        # We specify the user code for completion with model by default
        self._add_message("user", message=code)

        # First completion: do it changing the output format, i.e. using the fine-tuned model
        self._add_message("assistant", model=self.model_first)

        # Return the last message (the completion)
        return self.messages[-1]["content"]

    def feedback_step(self) -> bool:
        """
        Run the feedback step of the LLMCoder feedback loop

        Returns
        -------
        bool
            True if the completed code passes all the analyzers, False otherwise
        """
        # Construct the full code with the last two messages (i.e. the user code and the assistant code)
        completed_code = self.messages[-2]['content'] + self.messages[-1]['content']

        # Run the analyzers
        analyzer_results: list[dict] = []

        if self.feedback_variant == "separate":
            for analyzer in self.analyzers:
                analyzer_results.append(analyzer.analyze(completed_code))
        if self.feedback_variant == "coworker":
            raise NotImplementedError("Coworker feedback variant not implemented yet")

        # Check if all the analyzers passed
        if all([results['pass'] for results in analyzer_results]):
            # If all the analyzers passed, return True
            return True

        error_prompt = '\n'.join([results['message'] for results in analyzer_results if not results['pass']])

        self._add_message("user", message=error_prompt)
        self._add_message("assistant")

        # Add the analyzer results to the analyzer results list
        self._update_analyzer_pass_history(analyzer_results)

        return all([results['pass'] for results in analyzer_results])
