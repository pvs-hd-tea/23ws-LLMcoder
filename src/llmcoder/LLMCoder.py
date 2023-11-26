# First step in the prototype of completion fetching documentationimport openai
# TODO:
# 1. Modify structure prompt
# 2. Define alternative for feedback_variant
import os

import openai
from utils import get_openai_key

from .docanalyzer import APIDocumentationAnalyzer
from .synanalyzer import SyntaxAnalyzer
from .unittestanalyzer import UnitTestAnalyzer


class LLMCoder:
    # @ model_name string (e.g. "gpt-3.5-turbo")
    # @ feedback_invariant
        # "separate" -> execution of each analyzer separately
        # "coworker" -> analyzers interact with each other
    def __init__(self, model_first: str, model_feedback: str, feedback_variant: str, analyzers_list: list, system_prompt: str, max_iter: int):
        self.messages = []
        self.messages = self.add_message(self.messages, "system", system_prompt)
        self.model_first = model_first
        self.model_feedback = model_feedback

        if self.feedback_variant not in ["separate", "pipeline"]:
            raise ValueError("Inavlid feedback method")

        self.feedback_variant  = feedback_variant
        self.analyzers_list = analyzers_list
        self.client = openai.OpenAI(api_key = get_openai_key())
        self.iterations = 0
        self.max_iter = max_iter


    # Function for exception handling
    def check_reason(self, reason: str):
        try:
            if reason == "length":
                raise ValueError("Length output error.")

            if reason == "content_filter":
                raise ValueError("Omitted content.")

            if reason == "null":
                raise ValueError("API response still in progress.")

        except ValueError as ve:
            # Handle the exception specific to "reason" scenario
            print(f"Error: {ve}")

        except Exception as e:
            # Handle any other unexpected exceptions
            print(f"Unexpected Error: {e}")
        else:
            # Execute if no exception occurred
            print("No exception occurred.")


    # Function for completion
    # a) structured prompt with system and user roles
    # b) only system prompt
    def get_completion(self, code: str):
        # For each completion we will restablish the counter to 0
        self.iterations = 0


        # FIRST COMPLETION
        self.complete_first(code)
        self.iterations += 1


        # ITERATIVE FEEDBACK
        for i in range(self.max_iter):
            self.feedback_completion()

        return self.messages[-1]["content"]


    # Add messages to the prompt
    # 1. Add it to first completon as "user"
    # 2. Add it to feedcack as "assistant"
    # By default the model will be gpt-3.5-turbo, except first completion
    def add_message(self, messages: list[str], role: str, message: str | None = None, model: str = 'gpt-3.5-turbo') -> list[str]:
        # If the user is the assistant, generate a response
        if role == "assistant" and message is None:
            chat_completion = self.client.chat.completions.create(messages=messages, model=model)

            # Check if the procedure ended correctly
            fin_reason = chat_completion.choices[0].finish_reason

            # Attempt to call the function with "length" as the reason
            self.check_reason(fin_reason)

            message = chat_completion.choices[0].message.content

        # If role == "user" or "system"
        # Add the message to the messages list
        messages.append({
            "role": role,
            "content": message,
        })

        return messages

    # Method for first completion (no analysis)
    # "user" [system, user, code_to_complete (assistant)]
    # second iteration: [system, user, code_to_complete, "assistant", assistant_response]
    def complete_first(self, code: str):
        # We specify the user code for completion with model by default
        self.messages = self.add_message(self.messages, "user", code)
        # First completion: do it changing the output format, i.e. using the fine-tuned model
        self.messages = self.add_message(self.messages,  "asssistant", self.model_first)
        # The last message includes the completion of the assistant
        return self.messages[-1]


    # Method for feedback-based completion
    def feedback_completion(self) -> None:
        # messages[-2] = user code
        # messages[-1] = completed code
        completed_code = self.messages[-2]['content'] + self.messages[-1]['content']
        results = []

        if self.feedback_variant == "separate":
            for analyzer in self.analyzers_list:
                    results.append(analyzer.analyze(completed_code))
        if self.feedback_variant == "coworker":
            pass

        error_prompt = '\n'.join(results) + "\n"

        self.messages = self.add_message(self.messages, "user", error_prompt)
        self.messages = self.add_message(self.messages, "assistant")
