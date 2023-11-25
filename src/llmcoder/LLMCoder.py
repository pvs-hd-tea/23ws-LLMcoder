import openai

from llmcoder.utils import get_openai_key


class LLMCoder:
    def __init__(self, model_first: str, model_feedback: str, feedback_variant: str, analyzers_list: list, system_prompt: str, max_iter: int):
        self.messages: list = []
        self.messages = self.add_message(self.messages, "system", system_prompt)
        self.model_first = model_first
        self.model_feedback = model_feedback

        if feedback_variant not in ["separate", "coworker"]:
            raise ValueError("Inavlid feedback method")

        self.feedback_variant = feedback_variant
        self.analyzers_list = analyzers_list
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

        # Run the feedback loop until the code is correct or the max_iter is reached
        for i in range(self.max_iter):
            if self.feedback_completion():
                # If the feedback is correct, break the loop and return the code
                break

        # Return the last message
        return self.messages[-1]["content"]

    def add_message(self, messages: list[dict], role: str, message: str | None = None, model: str = 'gpt-3.5-turbo') -> list[dict]:
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
        self.messages = self.add_message(self.messages, "user", code)

        # First completion: do it changing the output format, i.e. using the fine-tuned model
        self.messages = self.add_message(self.messages, "asssistant", self.model_first)

        # Return the last message (the completion)
        return self.messages[-1]

    def feedback_completion(self) -> bool:
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

        error_prompt = '\n'.join([results['message'] for results in analyzer_results if not results['pass']])

        self.messages = self.add_message(self.messages, "user", error_prompt)
        self.messages = self.add_message(self.messages, "assistant")

        return all([results['pass'] for results in analyzer_results])
