# First step in the prototype of completion fetching documentationimport openai
# TODO:
# 1. Modify structure prompt
# 2. Define alternative for feedback_variant
import os
from .synanalyzer import SyntaxAnalyzer
from .docanalyzer import APIDocumentationAnalyzer
from .unittestanalyzer import UnitTestAnalyzer


class llmCoder:
    # @ model_name string (e.g. "gpt-3.5-turbo")
    # @ feedback_invariant
        # "separate" -> execution of each analyzer separately
        # "pipeline"
    def __init__(self, model_name: str, feedback_variant: str, analyzers_list: list):
        self.model_name = model_name
        self.feedback_variant = feedback_variant
        self.analyzers_list = analyzers_list

    def get_openai_api_key(self):
        openai.api_key = os.environ("OPENAI_API_KEY")
        return openai.api_key
    
    # Function for exception handling
    def check_reason(reason: str):
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
    def get_completion(self, prompt: str):

        # To be changed to:
        # messages=[{"role": "system", "content": system_msg},
        #                                 {"role": "user", "content": user_msg}]
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(model = self.model_name, messages = messages, temperature = 0)

        # Check if the procedure ended correctly
        fin_reason = response["choices"][0]["finish_reason"]
        # Example usage
        try:
            # Attempt to call the function with "length" as the reason
            check_reason(fin_reason)

        except Exception as e:
            # Handle any unexpected exception that might occur outside the function
            print(f"Unexpected Error: {e}")

        return response.choices[0].message["content"]

    # Method for first completion (no analysis)
    def complete_first(self, prompt: str):
        result = self.get_completion(prompt)
        return result
    
    # Method for feedback-based completion
    def feedback_completion(self, code_suggestion: str, prompt: str):
        try:
            if self.feedback_variant == "separate":
                # 1. Syntax analysis
                self.syntax_analysis(code_suggestion=code_suggestion)
                # 2. API Documentation analysis
                self.apidoc_analysis(code_suggestion=code_suggestion)

            if reason == "pipeline":
                pass

            else:
                raise ValueError("Incorrect feedback option.")

        except ValueError as ve:
            # Handle the exception specific to "reason" scenario
            print(f"Error: {ve}")

        

    def syntax_analysis(self, code_suggestion: str):
        syntaxanalyzer_instance = SyntaxAnalyzer(code_suggestion)
        return syntaxanalyzer_instance.analyze(code_suggestion = code_suggestion)
    
    def apidoc_analysis(self, code_suggestion: str):
        # try: - catch
        apianalyzer_instance = APIDocumentationAnalyzer(code_suggestion)
        return apianalyzer_instance.analyze(code_suggestion = code_suggestion)
    
    def unittest_analysis(self, code_suggestion: str):
        unittest_instance = UnitTestAnalyzer(code_suggestion)
        return unittest_instance.analyze(code_suggestion = code_suggestion)

    def quality_analysis(self, code_suggestion: str, prompt: str):
        pass
    

    

    




