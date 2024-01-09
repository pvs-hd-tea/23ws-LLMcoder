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


def agility_score(llmcoder_result: dict) -> float:
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
    n_analyzers_failed_each_loop = [sum([not results['pass'] for results in loop if results['type'] == "critical"]) for loop in analyzer_results_history]

    # The earlier all analyzers pass, the better
    # Compute a cumulative weighted sum of the number of analyzers passed each loop
    # Penalize later failures more
    agility_score = 0
    for i, n_analyzers_failed in enumerate(n_analyzers_failed_each_loop):
        agility_score += (i + 1) * n_analyzers_failed

    return agility_score
