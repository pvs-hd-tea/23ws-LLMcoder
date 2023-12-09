import numpy as np
from openai import OpenAI

from llmcoder.analyze.Analyzer import Analyzer
from llmcoder.utils import get_system_prompt, get_openai_key


class GPTScoreAnalyzer(Analyzer):
    def __init__(self, client: OpenAI | None = None, scoring_prompt: str | None = None, reduction: str | None = "geo"):
        """
        Create a new GPTScoreAnalyzer

        Parameters
        ----------
        client : openai.OpenAI
            The OpenAI client to use
        scoring_prompt : str
            The scoring prompt to use
        reduction : str | None, optional
            The reduction method to use, by default "geo"
        """
        self.client = client or OpenAI(api_key=get_openai_key())
        self.scoring_prompt = scoring_prompt or get_system_prompt("2023-12-09_Scorer_v1.1.txt")
        self.reduction = reduction

    def score_prompt(self, code_list: list[str]) -> str:
        """Concatenates the code snippets with the scoring prompt in the following format:

        Code snippet 1:
        ```python
        <code>
        ```

        Code snippet 2:
        ```python
        <code>
        ```
        ...

        Parameters
        ----------
        code_list : list[str]
            The list of code snippets to score

        Returns
        -------
        str
        """

        prompt = ""
        for i, code in enumerate(code_list):
            prompt += f"Code snippet {i + 1}:\n```python\n{code}\n```\n\n"

        return prompt

    def score_code(self, code: str | list[str]) -> float:
        """
        Score the provided code snippet() using the scoring model

        Parameters
        ----------
        code : str | list[str]
            The code snippet(s) to score
        client : openai.OpenAI
            The OpenAI client to use
        scoring_prompt : str
            The scoring prompt to use
        reduction : str | None, optional
            The reduction method to use, by default "geo"

        Returns
        -------
        float
            The score of the code snippet(s)
        """

        if isinstance(code, str):
            code = [code]

        messages = [
            {
                "role": "system",
                "content": self.scoring_prompt
            }, {
                "role": "user",
                "content": self.score_prompt(code)
            }
        ]
        completions = self.client.chat.completions.create(messages=messages, model="gpt-3.5-turbo", temperature=0)

        lines = completions.choices[0].message.content.split("\n")

        # Extract the scores from the response
        scores = []
        if "Code snippet" in lines[0]:
            for i, line in enumerate(lines):
                scores_for_snippet = []
                if line.startswith("Code snippet"):
                    for j in range(4):
                        try:
                            scores_for_snippet.append(float(lines[i + j + 1][lines[i + j + 1].index(":") + 1:]))
                        except ValueError:
                            print(f"[Scoring] Error while scoring code. Expected float, got: {completions.choices[0].message.content}")
                            scores_for_snippet.append(np.nan)
                    scores.append(scores_for_snippet)
        elif len(code) == 1:
            for i in range(4):
                try:
                    scores.append(float(lines[i][lines[i].index(":") + 1:]))
                except ValueError:
                    print(f"[Scoring] Error while scoring code. Expected float, got: {completions.choices[0].message.content}")
                    scores.append(np.nan)

        scores = np.atleast_2d(np.array(scores))

        match self.reduction:
            case "mean":
                return scores.mean(axis=1)
            case "max":
                return scores.max(axis=1)
            case "geo":
                return scores.prod(axis=1) ** (1 / scores.shape[1])
            case None:
                return scores
            case _:
                raise ValueError("Invalid reduction method")

    def analyze(self, input: str, completion: str, context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
        """
        Analyze the provided completion using the scoring model

        Parameters
        ----------
        input : str
            The input code
        completion : str
            The completion to analyze
        reduction : str | None, optional
            The reduction method to use, by default "geo"

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - pass: bool
                Whether the completion passed the analysis.
            - message: str
                The message returned by mypy.
        """
        if self.reduction is None:
            raise ValueError(f"Invalid reduction method: {self.reduction}. Must be one of: mean, max, geo")

        code = input + completion

        score = self.score_code(code)[0]

        return {
            "type": "score",
            "score": score,
            "pass": True,
            "message": ""
        }
