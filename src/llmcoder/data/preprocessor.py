import json
import os
import random
import re

import tiktoken
from tqdm import tqdm

from llmcoder.utils import get_data_dir, get_system_prompt


def count_lines(file_path: str) -> int:
    """
    Count the number of lines in a file.

    Parameters
    ----------
    file_path : str
        The path to the file.

    Returns
    -------
    int
        The number of lines in the file.
    """
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)


def get_file_contents(file_paths: list[str]) -> list[str]:
    """
    Get the contents of the given files.

    Parameters
    ----------
    file_paths : list[str]
        A list of paths to the files.

    Returns
    -------
    list[str]
        A list of the contents of the files.
    """
    contents = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            contents.append(file.read())

    return contents


def split_file(file_contents: str, min_pos: int = 1, max_pos: int = None) -> tuple[str, str]:
    """
    Split a file at a uniformly random position.

    Parameters
    ----------
    file_contents : str
        The contents of the file.
    min_pos : int, optional
        The minimum position to split the file at, by default 1
    max_pos : int, optional
        The maximum position to split the file at, by default None

    Returns
    -------
    tuple[str, str]
        A tuple containing the first and second part of the file.
    """
    if max_pos is None:
        max_pos = len(file_contents) - 1

    if min_pos > len(file_contents) - 1:
        raise ValueError("min_pos cannot be greater than the length of the file contents")

    if min_pos < 0:
        raise ValueError("min_pos cannot be negative")

    if max_pos > len(file_contents) - 1:
        raise ValueError("max_pos cannot be greater than the length of the file contents")

    if min_pos > max_pos:
        raise ValueError("min_pos cannot be greater than max_pos")

    split_pos = random.randint(min_pos, max_pos)

    return file_contents[:split_pos], file_contents[split_pos:]


def sample_files_from_dir(repo_dir: str, n_samples: int = 4, file_extensions: list[str] | None = None) -> list[str]:
    """
    Sample n_samples files from a repository based on the number of lines in each file.

    Parameters
    ----------
    repo_dir : str
        The path to the repository directory.
    n_samples : int, optional
        The number of files to sample, by default 4
    file_extensions : list[str], optional
        A list of file extensions to sample from, by default ['.py']

    Returns
    -------
    list[str]
        A list of paths to the sampled files.
    """
    if file_extensions is None:
        file_extensions = ['.py']

    # Check if the the files in the repository match the file extensions
    file_paths = [os.path.join(repo_dir, file) for file in os.listdir(repo_dir) if any(file.endswith(ext) for ext in file_extensions)]

    # Count the lines of each file
    line_counts = [count_lines(file_path) for file_path in file_paths]
    total_lines = sum(line_counts)

    if total_lines < n_samples:
        # Not enough lines to sample from in this repository
        return []

    # Sample n_samples files based on the number of lines in each file
    probabilities = [line_count / total_lines for line_count in line_counts]
    sampled_files = random.choices(file_paths, weights=probabilities, k=n_samples)

    return sampled_files


class Preprocessor:
    """
    A preprocessor for the fine-tuning data which samples files from scraped repositories, splits them into two parts and saves them in a format that can be used for fine-tuning.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    tokenizer : str, optional
        The tokenizer to use, by default "p50k_base" for gpt-3.5-turbo
    scraped_files_dir : str
        The directory to store the scraped files in, defaults to 'scraped_repos'.
    save_pairs_dir : str
        The directory to store the sampled files in, defaults to 'pairs'.
    save_data_dir : str
        The directory to store the preprocessed data in, defaults to 'github_mix'.
    system_prompt : str
        The system prompt to use, defaults to the default system prompt.
    disallowed_special_tokens : list[str]
        A list of disallowed special tokens, defaults to the default disallowed special tokens.
    """
    def __init__(self, dataset_name: str, tokenizer: str = "p50k_base", scraped_files_dir: str | None = None, save_pairs_dir: str | None = None, save_data_dir: str | None = None, system_prompt: str | None = None, disallowed_special_tokens: list[str] | None = None) -> None:
        self.name = dataset_name

        self.enc = tiktoken.get_encoding(tokenizer)

        if scraped_files_dir is None:
            self.scraped_files_dir = get_data_dir(self.name, "scraped_repos", create=True)  # /data/scraped_repos
        else:
            self.scraped_files_dir = scraped_files_dir

        if save_pairs_dir is None:
            self.save_pairs_dir = get_data_dir(self.name, "pairs", create=True)  # /data/pairs
        else:
            self.save_pairs_dir = save_pairs_dir

        if save_data_dir is None:
            self.save_data_dir = get_data_dir(self.name, create=True)  # /data/github_mix
        else:
            self.save_data_dir = save_data_dir

        if disallowed_special_tokens is None:
            # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L54
            ENDOFTEXT = "<|endoftext|>"
            FIM_PREFIX = "<|fim_prefix|>"
            FIM_MIDDLE = "<|fim_middle|>"
            FIM_SUFFIX = "<|fim_suffix|>"
            ENDOFPROMPT = "<|endofprompt|>"

            self.disallowed_special_tokens = [ENDOFTEXT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, ENDOFPROMPT]
        else:
            self.disallowed_special_tokens = disallowed_special_tokens

        if system_prompt is None:
            self.system_prompt = get_system_prompt()
        else:
            self.system_prompt = system_prompt

    def sample_files(self, n_samples: int = 4, file_extensions: list[str] | None = None) -> list[tuple[str, str]]:
        """
        Sample n_samples files from the scraped repositories, split them into two parts and return them.

        Parameters
        ----------
        n_samples : int, optional
            The number of files to sample, by default 4
        file_extensions : list[str], optional
            A list of file extensions to sample from, by default ['.py']

        Returns
        -------
        list[tuple[str, str]]
            A list of tuples containing the first and second part of the sampled files.
        """
        sampled_files_contents = []
        for repo_name in tqdm(os.listdir(self.scraped_files_dir)):
            if not os.path.isdir(os.path.join(self.scraped_files_dir, repo_name)):
                continue

            repo_dir = os.path.join(self.scraped_files_dir, repo_name)

            sampled_files = sample_files_from_dir(repo_dir, n_samples=n_samples, file_extensions=file_extensions)

            # Skip the repository if there are no files to sample from
            if sampled_files == []:
                print(f"Skipping {repo_name} because there are no files to sample from.")

            file_contents = get_file_contents(sampled_files)
            sampled_files_contents.extend(file_contents)

        # Split each file into two parts
        split_files_contents = [split_file(file_contents) for file_contents in sampled_files_contents]

        return split_files_contents

    def preprocess(self, split_files_contents: list[tuple[str, str]], min_input_len: int = 1_000, max_input_len: int = 10_000) -> list[tuple[str, str]]:
        """
        Truncate the input to the maximum length from the beginning of the file.

        Parameters
        ----------
        split_files_contents : list[tuple[str, str]]
            A list of tuples containing the first and second part of the sampled files.
        min_input_len : int, optional
            The minimum length of the input, by default 1_000
        max_input_len : int, optional
            The maximum length of the input, by default 10_000

        Returns
        -------
        list[tuple[str, str]]
            A list of tuples containing the first and second part of the sampled files.

        """
        truncated_split_files_contents = []

        for first_part, second_part in split_files_contents:
            random_max_input_len = random.randint(min_input_len, max_input_len)
            if len(first_part) > random_max_input_len:
                first_part = first_part[-random_max_input_len:]
            truncated_split_files_contents.append((first_part, second_part))

        # Warn the user that preprocessing is not over and that they whould manually truncate the output
        print("WARNING: Preprocessing is not over yet")
        print("Please manually truncate the output according to the following guidelines:")
        print("1. Consider the input and the information it provides.")
        print("2. Imagine what could possibly be inferred from the input.")
        print("3. Only keep the part of the output that is desirable but reasonably inferable from the input.")
        print("4. Run the formating method to format the output for OpenAI's API.")

        return truncated_split_files_contents

    def save_pairs(self, truncated_split_files_contents: list[tuple[str, str]]) -> None:
        """
        Save the data for manual truncation of the output.

        Parameters
        ----------
        truncated_split_files_contents : list[tuple[str, str]]
            A list of tuples containing the first and second part of the sampled files.
        """
        for i, (first_part, second_part) in enumerate(truncated_split_files_contents):

            pair_name = f"pair_{i}"
            pair_path = os.path.join(self.save_pairs_dir, pair_name)

            os.makedirs(pair_path, exist_ok=True)

            with open(os.path.join(pair_path, "input.txt"), 'w') as file:
                file.write(first_part)

            with open(os.path.join(pair_path, "output.txt"), 'w') as file:
                file.write(second_part)

        # Inform the user that the data is ready for manual truncation
        print(f"Data saved in {self.save_pairs_dir}.")
        print("Please manually truncate the output according to the following guidelines:")
        print("1. Consider the input and the information it provides.")
        print("2. Imagine what could possibly be inferred from the input.")
        print("3. Only keep the part of the output that is desirable but reasonably inferable from the input.")
        print("4. Run the formating method to format the output for OpenAI's API.")

    def build_conversations(self) -> list[list[dict]]:
        # Read the input and output files
        input_files = [os.path.join(self.save_pairs_dir, folder, "input.txt") for folder in os.listdir(self.save_pairs_dir)]
        output_files = [os.path.join(self.save_pairs_dir, folder, "output.txt") for folder in os.listdir(self.save_pairs_dir)]

        # Combine the system prompt, inout and output text into a conversation
        conversations = []
        for input_file, output_file in tqdm(zip(input_files, output_files)):
            with open(input_file, 'r') as file:
                input_text = file.read()

            with open(output_file, 'r') as file:
                output_text = file.read()

            conversation = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': input_text},
                {'role': 'assistant', 'content': output_text}
            ]
            conversations.append(conversation)

        # Remove disallowed special tokens from the conversations
        filtered_tokenized_conversations = []
        for conversation in tqdm(conversations):
            regex_pattern = "|".join([re.escape(token) for token in self.disallowed_special_tokens])
            filtered_tokenized_conversations.append([{k: re.sub(regex_pattern, "", v) for k, v in message.items()} for message in conversation])

        return filtered_tokenized_conversations

    def validate_conversations(self, conversations: list[list[dict]]) -> None:
        """
        Validate the conversations by tokenizing them.

        Parameters
        ----------
        conversations : list[list[dict]]
            A list of conversations.
        """
        # Try to tokenize the conversations
        # If the tokenization fails, remove the conversation
        tokenized_conversations = []
        for i, conversation in tqdm(enumerate(conversations)):
            try:
                tokenized_conversation = [{"role": message["role"], "tokenized_content": self.enc.encode(message["content"])} for message in conversation]
                tokenized_conversations.append(tokenized_conversation)
            except Exception as e:
                print(f"Failed to tokenize conversation. Skipping conversation {i}.")
                print(e)

    def save_conversations(self, conversations: list[list[dict]]) -> None:
        """
        Save the conversations in a jsonl file.

        Parameters
        ----------
        conversations : list[list[dict]]
            A list of conversations.
        """
        # Save the conversations in a jsonl file
        output_file = os.path.join(self.save_data_dir, "conversations.jsonl")
        with open(output_file, 'w') as file:
            for conversation in tqdm(conversations):
                file.write(json.dumps({"messages": conversation}, ensure_ascii=False) + '\n')

        # Inform the user that the data is fully processed and ready to be used in the OpenAI API
        print(f"Data saved in {output_file} and ready to be used in the OpenAI API.")
