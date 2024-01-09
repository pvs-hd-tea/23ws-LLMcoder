import json
import os


def read_data_from_pairs(pair_dir: str) -> list[tuple[str, str]]:
    """
    Reads data from a directory of pairs, e.g.

    /dataset
        /pair1
            input.txt
            output.txt
        /pair2
            input.txt
            output.txt
        ...

    Parameters
    ----------
    pair_dir : str
        Path to directory of pairs

    Returns
    -------
    list[tuple[str, str]]
        List of pairs of input and output
    """

    pairs = []
    for pair in os.listdir(pair_dir):

        pair_path = os.path.join(pair_dir, pair)
        input_path = os.path.join(pair_path, 'input.txt')
        output_path = os.path.join(pair_path, 'output.txt')

        with open(input_path, 'r') as f:
            input_text = f.read()
        with open(output_path, 'r') as f:
            output_text = f.read()

        pairs.append((input_text, output_text))

    return pairs


def read_data_from_conversations_file(conversations_file: str) -> list[tuple[str, str]]:
    """
    Reads data from a conversations file, e.g.

    /dataset
        /conversations.jsonl

    Parameters
    ----------
    conversations_file : str
        Path to conversations file

    Returns
    -------
    list[list[str]]
        List of conversations, where each conversation is a list of utterances
    """

    conversations = []
    with open(conversations_file, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))

    # Extract the user input and the target output from the conversations
    # The first message is a system prompt ("role" = "system"), the second message is the user input with "role" = "user" and the third message is the target output with "role" = "assistant"
    pairs = []
    for conversation in conversations:
        # Loop through the conversation until both the user input and the target output are found
        input_text = None
        output_text = None
        for message in conversation['messages']:
            if message['role'] == 'user':
                input_text = message['content']
            elif message['role'] == 'assistant':
                output_text = message['content']

            if input_text and output_text:
                break

        # If either the user input or the target output is missing, skip this conversation
        if not input_text or not output_text:
            continue

        pairs.append((input_text, output_text))

    return pairs


def construct_eval_conversation(pairs: list[tuple[str, str]], system_prompt: str) -> list[dict]:
    """
    Constructs a conversation for evaluation from a list of pairs,

    [
        {"role": "system", "content": <system_prompt>},
        {"role": "user", "content": <input1>},
        {"role": "assistant", "content": <output1>}
    ]

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        List of pairs of input and output
    system_prompt : str
        System prompt to use

    Returns
    -------
    list[dict]
        List of messages in the conversation
    """

    conversation = []

    # Add the system prompt
    conversation.append({
        'role': 'system',
        'content': system_prompt
    })

    # Add the user input and the target output
    for input_text, output_text in pairs:
        conversation.append({
            'role': 'user',
            'content': input_text
        })
        conversation.append({
            'role': 'assistant',
            'content': output_text
        })

    return conversation


def dump_results_to_json(results: dict[dict], output_file: str) -> None:
    """
    Dump the results to a JSON file

    Parameters
    ----------
    results : dict[dict]
        The results to dump
    output_file : str
        The path to the output file
    """

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def dump_results_to_readable(results: dict[dict], output_dir: str) -> None:
    """
    Dump the results to a human readable format.
    `results['log']` includes the captured verbose stdout from the LLMCoder run.
    Each message is written to a separate file in the output_dir based on the key in the results dict.

    Parameters
    ----------
    results : dict[dict]
        The results to dump
    output_dir : str
        The path to the output directory
    """

    os.makedirs(output_dir, exist_ok=True)
    for key, value in results.items():
        with open(os.path.join(output_dir, f'{key}.txt'), 'w') as f:
            f.write(value['log'])


def read_results_from_json(input_file: str) -> dict:
    """
    Read the results from a JSON file

    Parameters
    ----------
    input_file : str
        The path to the input file

    Returns
    -------
    dict
        The results
    """

    with open(input_file, 'r') as f:
        return json.load(f)
