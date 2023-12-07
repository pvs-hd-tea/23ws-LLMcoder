from difflib import SequenceMatcher  # , ndiff

import numpy as np
from Levenshtein import distance
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI
# from pyastsim.pyastsim import get_normed_content, get_pair_stats
from sentence_transformers import SentenceTransformer, util

from llmcoder.utils import get_openai_key

# from typing import Callable


def levenshtein_distance_score(ground_truth: str, llmcoder_result: dict | str) -> int:
    """
    Compute the Levenshtein distance between two strings.

    Parameters
    ----------
    ground_truth : str
        The first string to compare.
    llmcoder_result : dict | str
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    int
        The Levenshtein distance between the two strings.
    """
    if isinstance(llmcoder_result, dict):
        completion = llmcoder_result['messages'][-1]['content']
    else:
        completion = llmcoder_result

    return distance(ground_truth, completion)


def bleu_score(ground_truth: str | list[str], llmcoder_result: dict | str) -> float:
    """
    Compute the BLEU score between a candidate and a list of references.

    Parameters
    ----------
    references : str | list[str]
        The reference string(s) to compare to the candidate.
    llmcoder_result : dict | str
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    float
        The BLEU score between the candidate and the references.
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    if isinstance(llmcoder_result, dict):
        completion = llmcoder_result['messages'][-1]['content']
    else:
        completion = llmcoder_result

    return sentence_bleu(ground_truth, completion)


def trf_similarity_score(ground_truth: str, llmcoder_result: dict | str, model: str = "sentence-transformers/all-roberta-large-v1") -> float:
    """
    Compute the sentence similarity between two strings with a transformer model.

    Parameters
    ----------
    ground_truth : str
        The first string to compare.
    llmcoder_result : dict | str
        The llmcoder result containing the conversation (the last message is the completion).
    model : str, optional
        The name of the transformer model to use, by default "all-roberta-large-v1 "

    Returns
    -------
    float
        The sentence similarity between the two strings.
    """

    model = SentenceTransformer(model)
    gt_embedding = model.encode(ground_truth)

    if isinstance(llmcoder_result, dict):
        completion = llmcoder_result['messages'][-1]['content']
    else:
        completion = llmcoder_result

    completion_embedding = model.encode(completion)
    return util.pytorch_cos_sim(gt_embedding, completion_embedding).item()


# FIXME: This does not work properly since the code snippets are not full python programs I think
# def _ast_compare_strings(string1: str, string2: str, threshold: int = 80, show_diff: bool = False, function: Callable = None) -> float:
#     """
#     Compare two strings with ASTs.

#     Parameters
#     ----------
#     string1 : str
#         The first string to compare.
#     string2 : str
#         The second string to compare.
#     threshold : int, optional
#         The similarity threshold, by default 80
#     show_diff : bool, optional
#         Whether to show the diff between the two strings, by default False
#     function : function, optional
#         The function to use to parse the ASTs, by default None

#     Returns
#     -------
#     float
#         The similarity between the two strings.
#     """
#     # Normalize the content of the strings
#     submissions = [get_normed_content(s, function) for s in [string1, string2]]

#     # Get the similarity and edit distance between the two strings
#     pair_stats = get_pair_stats(submissions)

#     # Check if the similarity is above the threshold
#     if pair_stats[0] > threshold:
#         print(f"Detected similarity of {int(pair_stats[0])}% with edit distance of {pair_stats[1]}")

#         # If show_diff is True, print the diff
#         if show_diff:
#             print('\n'.join(ndiff(submissions[0][1].splitlines(), submissions[1][1].splitlines())))

#         return pair_stats[0]

#     return 0


# def ast_similarity_score(ground_truth: str, completion: str) -> float:
#     """
#     Compute the similarity between two python code snippets with ASTs.

#     Parameters
#     ----------
#     ground_truth : str
#         The first string to compare.
#     completion : str
#         The second string to compare.

#     Returns
#     -------
#     float
#         The similarity between the two strings.
#     """
#     return _ast_compare_strings(ground_truth, completion)


def sequence_matcher_score(ground_truth: str, llmcoder_result: dict | str) -> float:
    """
    Compute the similarity between two strings with the SequenceMatcher algorithm.

    Parameters
    ----------
    ground_truth : str
        The first string to compare.
    llmcoder_result : dict | str
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    float
        The similarity between the two strings.
    """
    if isinstance(llmcoder_result, dict):
        completion = llmcoder_result['messages'][-1]['content']
    else:
        completion = llmcoder_result

    return SequenceMatcher(None, ground_truth, completion).ratio()


def _user_prompt_templste(code_1: str, code_2: str, qualities_list: list[str]) -> str:
    quality_list_string = '\n'.join([f'- {q}' for q in qualities_list])
    return f"""Assess and compare these two code snippets and evaluate the completions. Do your own analysis and also prip the following criteria:
{quality_list_string}

CODE 1:
```python
{code_1}
```

CODE 2:
```python
{code_2}
```"""


def gpt_reviewer_score(ground_truth: str, llmcoder_result: dict | str, model: str = "gpt-3.5-turbo", qualities_list: list[str] | None = None, max_iter: int = 5) -> float:
    """
    Compute the similarity of qualities of two strings with GPT-3.

    Parameters
    ----------
    ground_truth : str
        The first string to compare.
    llmcoder_result : dict
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    float
        The similarity between the two strings. Positive if the completion is better than the ground truth, negative otherwise.
    """
    client = OpenAI(api_key=get_openai_key())

    system_prompt_compare = """You are a data scientist tasked with comparing and evaluating code completions made by a language model.
The user will submit two code snippets with the same beginning but different completions.
Given these snippets, you evaluate the completions in a concise way, and give each a score between 0 and 10, with 0 being the worst (unusable completion that would make a developer frustrated) and 10 being the best (perfect completion that would make a developer happy).
The user may ask you to prioritize different qualities of the code.
Take these priorities into account when scoring the completions.
Your output must always have the following format:
```
COMPARISON:
<comparison of the two completions with regard to the requested qualities>

SCORE 1: <score for completion 1, integer between 0 and 10>
SCORE 2: <score for completion 2, integer between 0 and 10>
```

Do not include any other information in your output.
It is very important that the output following "SCORE 1: " and "SCORE 2: " is a single integer between 0 and 10, with no other characters or spaces since scores will later be parsed at this exact location.
Therefore, make sure to keep your comparison (the text after COMPARISON:) concise, and adhere to the score format exactly.
"""

    if qualities_list is None:
        qualities_list = [
            "alignment of the completion with the given code (hint: the given code is the same for both completions)",
            "correctness",
            "plausibility",
            "readability",
            "efficiency"
        ]

    if isinstance(llmcoder_result, dict):
        completion = llmcoder_result['messages'][-1]['content']
    else:
        completion = llmcoder_result

    user_prompt = _user_prompt_templste(ground_truth, completion, qualities_list)

    messages = [
        {
            "role": "system",
            "content": system_prompt_compare
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    scores = []

    for _ in range(max_iter):

        chat_completion = client.chat.completions.create(messages=messages, model=model, temperature=0.2)  # type: ignore
        message = chat_completion.choices[0].message.content

        if message is None:
            return np.nan

        # Get the scores from the messag with regex (the numbers that follow "SCORE 1: " and "SCORE 2: ")
        scores = [float(s.split(": ")[1]) for s in message.split("\n") if s.startswith("SCORE")]

        # If both scores are recognized, break
        if len(scores) == 2:
            return scores[1] - scores[0]

    print(f"WARN: Could not parse scores from message: {message}")
    return np.nan
