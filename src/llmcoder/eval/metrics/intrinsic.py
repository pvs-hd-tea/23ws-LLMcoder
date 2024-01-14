import numpy as np
import tiktoken


def loops_required_score(llmcoder_result: dict) -> int:
    """
    Compute the number of loops required to generate the ground truth from the llmcoder result.

    Parameters
    ----------
    llmcoder_result : dict
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    int
        The number of loops required to generate the ground truth from the llmcoder result.
    """
    # Subtract the system prompt, the first user message and the first completion
    # Then divide by two because each loop has two messages
    return (len(llmcoder_result['messages']) - 3) // 2


def tokens_used_score(llmcoder_result: dict, tokenizer_name: str = "p50k_base") -> int:
    """
    Compute the number of tokens used to generate the ground truth from the llmcoder result.

    Parameters
    ----------
    llmcoder_result : dict
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    int
        The number of tokens used to generate the ground truth from the llmcoder result.
    """
    enc = tiktoken.get_encoding(tokenizer_name)

    # For each message in the conversation count the tokens of the content
    n_tokens = 0
    for message in llmcoder_result['messages']:
        n_tokens += len(enc.encode(message['content']))

    return n_tokens


def agility_score(llmcoder_result: dict, length_scale: float = 1) -> float:
    """
    Compute the agility score of the llmcoder result.

    Parameters
    ----------
    llmcoder_result : dict
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    float
        The agility score of the llmcoder result.
    """
    # Read the analyzer pass history from the llmcoder result
    analyzer_results_history: list[dict[str, dict[str, bool]]] = llmcoder_result['analyzer_results']

    # For each loop, check how many analyzers passed
    scores_each_loop = np.array([sum([analyzer_results['score'] for analyzer_name, analyzer_results in iteration_results.items()]) for iteration_results in analyzer_results_history])

    # The earlier the scores improve, the better
    weights = np.exp(- np.arange(len(scores_each_loop) - 1) * length_scale)
    return np.sum(weights * np.sign(np.diff(scores_each_loop)) * np.diff(scores_each_loop)**2)


def time_score(llmcoder_result: dict) -> float:
    """
    Return the time of the llmcoder result.

    Parameters
    ----------
    llmcoder_result : dict
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    float
        The time it took to generate the result.
    """
    return llmcoder_result['time']


def all_analyzers_passed_score(llmcoder_result: dict) -> bool:
    """
    Return whether all analyzers passed.

    Parameters
    ----------
    llmcoder_result : dict
        The llmcoder result containing the conversation (the last message is the completion).

    Returns
    -------
    bool
        Whether all analyzers passed.
    """
    # Read the analyzer pass history from the llmcoder result
    analyzer_results_history: list[dict[str, dict[str, bool]]] = llmcoder_result['analyzer_results']

    # Check whether all analyzers passed in the last iteration
    return all([analyzer_results['pass'] for analyzer_name, analyzer_results in analyzer_results_history[-1].items() if analyzer_results['type'] == 'critical'])
