import openai

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.utils import get_openai_key, get_system_prompt


class LLMCoder:
    def __init__(self,
                 analyzers: list[str] = ["api_documentation_analyzer_v1"],
                 model_first: str = "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d",
                 model_feedback: str = "gpt-3.5-turbo",
                 feedback_variant: str = "separate",
                 system_prompt: str | None = None,
                 max_iter: int = 10):
        """
        Initialize the LLMCoder

        Parameters
        ----------
        analyzers : list[str], optional
            The list of analyzers to use, by default ["api_documentation_analyzer_v1"]
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
        """
        self.messages: list = []

        if system_prompt is None:
            system_prompt = get_system_prompt()

        self.messages = self._add_message(self.messages, "system", system_prompt)

        self.model_first = model_first
        self.model_feedback = model_feedback

        if feedback_variant not in ["separate", "coworker"]:
            raise ValueError("Inavlid feedback method")

        self.feedback_variant = feedback_variant
        self.analyzers_list = [AnalyzerFactory.create_analyzer(analyzer) for analyzer in analyzers]
        self.client = openai.OpenAI(api_key=get_openai_key())
        self.iterations = 0
        self.max_iter = max_iter

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

        if len(self.analyzers_list) > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for _ in range(self.max_iter):
                if self.feedback_step():
                    # If the feedback is correct, break the loop and return the code
                    break

        # Return the last message
        return self.messages[-1]["content"]

    def _add_message(self, messages: list[dict], role: str, message: str | None = None, model: str = 'gpt-3.5-turbo') -> list[dict]:
        # If the user is the assistant, generate a response
        if role == "assistant" and message is None:
            chat_completion = self.client.chat.completions.create(messages=messages, model=model)  # type: ignore

            message = chat_completion.choices[0].message.content

        messages.append({
            "role": role,
            "content": message,
        })

        return messages

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
        self.iterations = 0

        # We specify the user code for completion with model by default
        self.messages = self._add_message(self.messages, "user", code)

        # First completion: do it changing the output format, i.e. using the fine-tuned model
        self.messages = self._add_message(self.messages, "asssistant", self.model_first)

        # Return the last message (the completion)
        return self.messages[-1]

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
            for analyzer in self.analyzers_list:
                analyzer_results.append(analyzer.analyze(completed_code))
        if self.feedback_variant == "coworker":
            raise NotImplementedError("Coworker feedback variant not implemented yet")

        # Check if all the analyzers passed
        if all([results['pass'] for results in analyzer_results]):
            # If all the analyzers passed, return True
            return True

        error_prompt = '\n'.join([results['message'] for results in analyzer_results if not results['pass']])

        self.messages = self._add_message(self.messages, "user", error_prompt)
        self.messages = self._add_message(self.messages, "assistant")

        return all([results['pass'] for results in analyzer_results])
