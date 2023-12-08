import json
import os
from datetime import datetime

import openai
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        self.analyzer_pass_history: list[list[dict]] = []
        self.analyzer_message_history: list[list[dict]] = []
        self.max_iter = max_iter
        self.messages: list = []

        # Set up the analyzers
        if analyzers is None:
            self.analyzers = []
        else:
            self.analyzers = [AnalyzerFactory.create_analyzer(analyzer) for analyzer in analyzers]

        # Set up the scoring model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)

        self.completion_score_tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base")
        self.completion_score_model = AutoModelForSequenceClassification.from_pretrained("microsoft/CodeBERT-base").to(self.device)

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

        # Add the system prompt to the messages
        self._add_message("system", message=self.system_prompt)

    def to(self, device: str | torch.device) -> None:
        """
        Move the scoring model to the specified device

        Parameters
        ----------
        device : str | torch.device
            The device to use
        """
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError("Invalid device type")
        self.completion_score_model = self.completion_score_model.to(self.device)

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
        print("Creating first completion...")
        self.complete_first(code, temperature, n)

        print("Starting feedback loop...")
        if len(self.analyzers) > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for i in range(self.max_iter):
                print(f"Starting feedback iteration {i + 1}...")
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

            messages = [choice.message.content for choice in chat_completions.choices]

            # Get the best message according to the scoring model
            scores = self.completion_score_tokenizer(messages, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                scores = self.completion_score_model(**scores.to(self.device)).logits[:, 0].cpu().numpy()  # The first class captures the code quality, the second one the need for comments (not used here))

            # Get the best message according to the scoring model
            message = messages[scores.argmax()]

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
        self.analyzer_message_history = []

    def _update_analyzer_history(self, analyzer_results: list[dict]) -> None:
        """
        Add the analyzer results to the analyzer results list

        Parameters
        ----------
        analyzer_results : list[dict]
            The analyzer results to add
        """
        self.iterations += 1
        self.analyzer_pass_history.append([results['pass'] for results in analyzer_results])
        self.analyzer_message_history.append([results['message'] for results in analyzer_results])

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
        analyzer_results: list[dict] = []

        if self.feedback_variant == "separate":
            print("Analyzing code...")
            for analyzer in self.analyzers:
                print(f"Running {analyzer.__class__.__name__}...")
                analyzer_results.append(analyzer.analyze(self.messages[1]['content'], self.messages[-1]['content']))
        if self.feedback_variant == "coworker":
            raise NotImplementedError("Coworker feedback variant not implemented yet")

        # Print how many analyzers have passed
        print(f"{sum([results['pass'] for results in analyzer_results if results['pass'] is not None])} / {len(analyzer_results)} analyzers passed")

        # Add the analyzer results to the analyzer results list
        self._update_analyzer_history(analyzer_results)

        # Check if all the analyzers passed
        if all([results['pass'] for results in analyzer_results if results['pass'] is not None]):
            # If all the analyzers passed, return True
            return True

        error_prompt = '[INST]\nConsider the following in your next completion:\n[ANALYSIS]\n' + '\n'.join([results['message'] for results in analyzer_results if not results['pass']]) + '\n[/ANALYSIS]\nSeamlessly complete the following code:\n[/INST]\n'

        self._add_message("user", message=error_prompt + self.messages[1]['content'])
        self._add_message("assistant", model=self.model_first, temperature=temperature, n=n)  # model_first works quite good here

        return False
