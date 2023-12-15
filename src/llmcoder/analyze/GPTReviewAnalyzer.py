import os
import re

from openai import OpenAI

from llmcoder.analyze.Analyzer import Analyzer
from llmcoder.utils import get_openai_key, get_system_prompt, get_system_prompt_dir


class GPTReviewAnalyzer_v1(Analyzer):
    """
    Concise, professional Python code advisor for targeted feedback.

    Parameters
    ----------
    model : str, optional
        The model to use, by default "gpt-3.5-turbo"
    system_prompt : str, optional
        The system prompt to use, by default `2023-12-02_GPTReviewAnalyzer_v4.txt`, created by GPTBuilder
    min_score : int, optional
        The minimum score to pass the review, by default 6

    """
    def __init__(self, model: str = "gpt-3.5-turbo", system_prompt: str | None = None, min_score: int = 6, temperature: float = 0.2):
        """
        Initialize the GPTReviewAnalyzer

        Parameters
        ----------
        model : str, optional
            The model to use, by default "gpt-3.5-turbo"
        system_prompt : str, optional
            The system prompt to use, by default `2023-12-02_GPTReviewAnalyzer_v4.txt`, created by GPTBuilder
        min_score : int, optional
            The minimum score to pass the review, by default 6
        """
        self.model = model

        self.messages: list[dict[str, str]] = []

        self.client = OpenAI(api_key=get_openai_key())

        if system_prompt is None:
            self.system_prompt = get_system_prompt("2023-12-02_GPTReviewAnalyzer_v4.txt")
        elif system_prompt in os.listdir(get_system_prompt_dir()):
            self.system_prompt = get_system_prompt(system_prompt)
        else:
            self.system_prompt = system_prompt

        self.min_score = min_score

        self.temperature = temperature

    def analyze(self, input: str, completion: str, context: dict[str, dict[str, bool | str]] | None = None) -> dict:
        """
        Analyze the input and completion

        Parameters
        ----------
        input : str
            The input to analyze
        completion : str
            The completion to analyze

        Returns
        -------
        dict
            The analysis of the input and completion

        """
        self.messages = []  # Think about it having a memory in the future and running in parallel as a discriminative reviewer with the generative LLMcoder

        self.messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        self.messages.append({
            "role": "user",
            "content": self._prompt_template(input, completion)
        })
        chat_completion = self.client.chat.completions.create(messages=self.messages, model=self.model, temperature=self.temperature) # type: ignore

        self.messages.append({
            "role": "assistant",
            "content": str(chat_completion.choices[0].message.content)
        })

        # Find the score in the message. The score line has the format "SCORE: <score, int{0, 10}>"
        # Define a robust regex to find the score
        score_regex = re.compile(r"SCORE: (\d+)")
        score_match = score_regex.search(self.messages[-1]["content"])

        if score_match is None:
            return {
                "message": self.messages[-1]["content"],
                "pass": None
            }

        return {
            "message": self.messages[-1]["content"],
            "pass": int(score_match.group(1)) >= self.min_score
        }

    @staticmethod
    def _prompt_template(input: str, completion: str) -> str:
        return f"""[INCOMPLETE]{input}[/INCOMPLETE]

[COMPLETE]{input + completion}[/COMPLETE]"""
