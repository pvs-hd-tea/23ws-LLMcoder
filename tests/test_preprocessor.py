from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest

from llmcoder.data.preprocessor import FineTunePreprocessor, count_lines, get_file_contents, sample_files_from_dir, split_file


# Generated with GPT-4 under supervision
# count_lines
def test_count_lines_normal() -> None:
    # Mock a file with 5 lines
    mock_file = mock_open(read_data='line1\nline2\nline3\nline4\nline5')
    with patch('builtins.open', mock_file):
        assert count_lines('any_file.txt') == 5


def test_count_lines_empty() -> None:
    # Mock an empty file
    mock_file = mock_open(read_data='')
    with patch('builtins.open', mock_file):
        assert count_lines('empty_file.txt') == 0


def test_count_lines_nonexistent() -> None:
    # Test for a file that doesn't exist. No need to mock here.
    with pytest.raises(FileNotFoundError):
        count_lines('nonexistent_file.txt')


# get_file_contents
def test_get_file_contents_multiple_files() -> None:
    # Mocking multiple files with different contents
    mock_files = {
        'file1.txt': 'Content of file 1',
        'file2.txt': 'Content of file 2'
    }

    def mock_file_open(file: Mock, mode: Mock, encoding: str | None = None) -> str:
        if file in mock_files:
            return mock_open(read_data=mock_files[file]).return_value
        else:
            raise FileNotFoundError

    with patch('builtins.open', side_effect=mock_file_open):
        result = get_file_contents(['file1.txt', 'file2.txt'])
        assert result == ['Content of file 1', 'Content of file 2']


def test_get_file_contents_empty_file() -> None:
    # Mocking an empty file
    mock_file = mock_open(read_data='')
    with patch('builtins.open', mock_file):
        assert get_file_contents(['empty_file.txt']) == ['']


def test_get_file_contents_nonexistent() -> None:
    # Testing for a file that doesn't exist. No need to mock here.
    with pytest.raises(FileNotFoundError):
        get_file_contents(['nonexistent_file.txt'])


# split_file
def test_split_file_normal() -> None:
    # Mock the random.randint function to return a fixed value
    with patch('random.randint', return_value=5):
        part1, part2 = split_file("abcdefgh")
        assert part1 == "abcde"
        assert part2 == "fgh"


def test_split_file_min_max() -> None:
    with patch('random.randint', return_value=7):
        part1, part2 = split_file("abcdefgh", min_pos=3, max_pos=7)
        assert part1 == "abcdefg"
        assert part2 == "h"


def test_split_file_min_greater_than_length() -> None:
    with pytest.raises(ValueError):
        split_file("abc", min_pos=4)


def test_split_file_max_greater_than_length() -> None:
    with pytest.raises(ValueError):
        split_file("abc", max_pos=4)


def test_split_file_min_greater_than_max() -> None:
    with pytest.raises(ValueError):
        split_file("abc", min_pos=5, max_pos=4)


def test_split_file_min_equal_to_max() -> None:
    # This should work
    with patch('random.randint', return_value=5):
        part1, part2 = split_file("abcdefgh", min_pos=5, max_pos=5)
        assert part1 == "abcde"
        assert part2 == "fgh"


def test_split_file_min_less_than_zero() -> None:
    with pytest.raises(ValueError):
        split_file("abc", min_pos=-1)


# sample_files_from_dir
def mock_count_lines(file_path: str) -> int:
    # Mock behavior of count_lines function
    return len(file_path)  # Example behavior, you can adjust this


@patch('os.listdir')
@patch('os.path.join')
@patch('llmcoder.data.preprocessor.count_lines', side_effect=mock_count_lines)
@patch('random.choices')
def test_sample_files_from_dir_normal(mock_choices: MagicMock, mock_count_lines: MagicMock, mock_join: MagicMock, mock_listdir: MagicMock) -> None:
    # Setting up the mock environment
    mock_listdir.return_value = ['file1.py', 'file2.py', 'file3.txt']
    mock_join.side_effect = lambda repo_dir, file: f"{repo_dir}/{file}"
    mock_choices.side_effect = lambda file_paths, weights, k: file_paths[:k]

    repo_dir = "/path/to/repo"
    n_samples = 2
    file_extensions = ['.py']

    expected_files = ['/path/to/repo/file1.py', '/path/to/repo/file2.py']
    sampled_files = sample_files_from_dir(repo_dir, n_samples, file_extensions)
    assert sampled_files == expected_files


def test_fine_tune_preprocessor_initialization() -> None:
    # Test with specific parameters
    custom_scraped_files_dir = "/custom/scraped"
    custom_save_pairs_dir = "/custom/save/pairs"
    custom_system_prompt = "Custom system prompt"
    custom_disallowed_tokens = ["TOKEN1", "TOKEN2"]

    preprocessor_custom = FineTunePreprocessor(dataset_name='pytest',
                                               scraped_files_dir=custom_scraped_files_dir,
                                               save_pairs_dir=custom_save_pairs_dir,
                                               system_prompt=custom_system_prompt,
                                               disallowed_special_tokens=custom_disallowed_tokens)

    assert preprocessor_custom.scraped_files_dir == custom_scraped_files_dir
    assert preprocessor_custom.save_pairs_dir == custom_save_pairs_dir
    assert preprocessor_custom.system_prompt == custom_system_prompt
    assert preprocessor_custom.disallowed_special_tokens == custom_disallowed_tokens


@patch('os.listdir')
@patch('os.path.join', side_effect=lambda *args: "/".join(args))
@patch('llmcoder.data.preprocessor.sample_files_from_dir')
@patch('llmcoder.data.preprocessor.get_file_contents')
@patch('llmcoder.data.preprocessor.split_file')
@patch('tqdm.tqdm', side_effect=lambda x: x)
@patch('os.path.isdir', return_value=True)
def test_sample_files(mock_os_path_isdir: MagicMock,
                      mock_tqdm: MagicMock,
                      mock_split_file: MagicMock,
                      mock_get_file_contents: MagicMock,
                      mock_sample_files_from_dir: MagicMock,
                      mock_join: MagicMock,
                      mock_listdir: MagicMock) -> None:
    # Setup the mocks
    mock_listdir.return_value = ['repo1', 'repo2']
    mock_sample_files_from_dir.side_effect = lambda repo_dir, **kwargs: [f"{repo_dir}/file1.py", f"{repo_dir}/file2.py"]
    mock_get_file_contents.side_effect = lambda file_paths: [f"Contents of {path}" for path in file_paths]
    mock_split_file.side_effect = lambda contents: (contents[:5], contents[5:])

    # Initialize FineTuner instance
    preprocessor = FineTunePreprocessor(dataset_name='pytest', scraped_files_dir="/mocked/data/dir/scraped_repos")
    result = preprocessor.sample_files()

    # Assertions
    assert len(result) == 4  # 2 files per repository, 2 repositories
    for first_part, second_part in result:
        assert first_part.startswith("Conte")
        assert second_part.startswith("nts of ")


@patch('random.randint', return_value=5000)
def test_preprocess(mock_randint: MagicMock) -> None:
    # Initialize FineTuner instance
    preprocessor = FineTunePreprocessor(dataset_name='pytest')  # Adjust based on how your class is initialized

    # Simulated split file contents
    split_files_contents = [("A" * 6000, "B" * 3000), ("C" * 4000, "D" * 2000)]

    # Process the contents
    processed_contents = preprocessor.preprocess(split_files_contents)

    # Assertions
    for first_part, _ in processed_contents:
        assert len(first_part) <= 5000  # Check if the first part is correctly truncated

    # Check specific content lengths
    assert len(processed_contents[0][0]) == 5000  # First content should be truncated
    assert len(processed_contents[1][0]) == 4000  # Second content should remain unchanged

    # Check content integrity
    assert processed_contents[0][0].startswith("A" * 1000)  # First 1000 characters should be 'A's
    assert processed_contents[1][0].startswith("C" * 4000)  # Should remain all 'C's


@patch('os.makedirs')
@patch('os.path.join', side_effect=lambda *args: "/".join(args))
@patch('builtins.open', new_callable=mock_open, read_data="")
def test_save_pairs(mock_file_open: MagicMock, mock_path_join: MagicMock, mock_makedirs: MagicMock) -> None:
    # Initialize FineTunePreprocessor instance
    preprocessor = FineTunePreprocessor(dataset_name='pytest', save_pairs_dir="/mocked/data/dir/fine-tune-pairs")

    # Sample data to test
    truncated_split_files_contents = [("input1", "output1"), ("input2", "output2")]

    # Call save_pairs
    preprocessor.save_pairs(truncated_split_files_contents)

    # Verify directory creation
    expected_dirs = [f"/mocked/data/dir/fine-tune-pairs/pair_{i}" for i in range(len(truncated_split_files_contents))]
    mock_makedirs.assert_has_calls([call(dir_path, exist_ok=True) for dir_path in expected_dirs], any_order=True)

    # Verify file creation and contents
    expected_file_calls = []
    for i, (input_content, output_content) in enumerate(truncated_split_files_contents):
        input_file_path = f"/mocked/data/dir/fine-tune-pairs/pair_{i}/input.txt"
        output_file_path = f"/mocked/data/dir/fine-tune-pairs/pair_{i}/output.txt"
        expected_file_calls.extend([
            call(input_file_path, 'w', encoding='iso-8859-1'),
            call().__enter__(),
            call().write(input_content),
            call().__exit__(None, None, None),
            call(output_file_path, 'w', encoding='iso-8859-1'),
            call().__enter__(),
            call().write(output_content),
            call().__exit__(None, None, None)
        ])

    mock_file_open.assert_has_calls(expected_file_calls, any_order=True)


class TestFineTunePreprocessor(TestCase):
    @patch('os.listdir', return_value=['pair_0', 'pair_1'])
    @patch('os.path.join', side_effect=lambda *args: "/".join(args))
    @patch('builtins.open', new_callable=mock_open, read_data="Sample Text")
    @patch('tqdm.tqdm', side_effect=lambda x: x)
    def test_build_conversations(self, mock_tqdm: MagicMock, mock_file_open: MagicMock, mock_path_join: MagicMock, mock_listdir: MagicMock) -> None:
        # Initialize FineTunePreprocessor instance
        system_prompt = "System prompt text"
        disallowed_tokens = ["disallowed_token1", "disallowed_token2"]
        preprocessor = FineTunePreprocessor(dataset_name='pytest', system_prompt=system_prompt, disallowed_special_tokens=disallowed_tokens)

        # Call build_conversations
        conversations = preprocessor.build_conversations()

        # Verify file reading
        expected_file_calls = []
        for i in range(2):  # Assuming you have two pairs of files
            input_file_path = f"{preprocessor.save_pairs_dir}/pair_{i}/input.txt"
            output_file_path = f"{preprocessor.save_pairs_dir}/pair_{i}/output.txt"
            expected_file_calls.extend([
                call(input_file_path, 'r', encoding='iso-8859-1'),
                call().__enter__(),
                call().read(),
                call().__exit__(None, None, None),
                call(output_file_path, 'r', encoding='iso-8859-1'),
                call().__enter__(),
                call().read(),
                call().__exit__(None, None, None),
            ])

        mock_file_open.assert_has_calls(expected_file_calls, any_order=True)

        # Verify the structure and content of conversations
        self.assertEqual(len(conversations), 2)  # Two conversations
        for conversation in conversations:
            self.assertEqual(len(conversation), 3)  # Three messages per conversation
            self.assertIn({'role': 'system', 'content': system_prompt}, conversation)
            self.assertIn({'role': 'user', 'content': "Sample Text"}, conversation)
            self.assertIn({'role': 'assistant', 'content': "Sample Text"}, conversation)

            # Check for absence of disallowed tokens
            for message in conversation:
                self.assertNotIn(disallowed_tokens[0], message['content'])
                self.assertNotIn(disallowed_tokens[1], message['content'])

    @patch('tqdm.tqdm', side_effect=lambda x: x)
    @patch('tiktoken.get_encoding')
    def test_validate_conversations(self, mock_get_encoding: MagicMock, mock_tqdm: MagicMock) -> None:
        # Mock the encoding object
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        # Mock the encode method on the encoder
        def mock_encode(content: str) -> str:
            if content == 'Goodbye':
                raise Exception("Tokenization failed")
            return f"tokenized_{content}"
        mock_encoder.encode.side_effect = mock_encode

        # Initialize FineTunePreprocessor instance
        system_prompt = "System prompt text"
        disallowed_tokens = ["disallowed_token1", "disallowed_token2"]
        preprocessor = FineTunePreprocessor(dataset_name='pytest', system_prompt=system_prompt, disallowed_special_tokens=disallowed_tokens)

        # Sample conversations
        conversations = [
            [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}],
            [{'role': 'user', 'content': 'Goodbye'}, {'role': 'assistant', 'content': 'Bye'}]
        ]

        # Call validate_conversations
        preprocessor.validate_conversations(conversations)

        # Check the results
        expected_number_of_calls = 3  # 4 messages but one of them is invalid (Goodbye)
        self.assertEqual(mock_encoder.encode.call_count, expected_number_of_calls)

    @patch('os.path.join', side_effect=lambda *args: "/".join(args))
    @patch('builtins.open', new_callable=mock_open, read_data="")
    @patch('tqdm.tqdm', side_effect=lambda x: x)
    @patch('json.dumps')
    def test_save_conversations(self, mock_json_dumps: MagicMock, mock_tqdm: MagicMock, mock_file_open: MagicMock, mock_path_join: MagicMock) -> None:
        # Initialize FineTunePreprocessor instance
        fine_tuner = FineTunePreprocessor(dataset_name='pytest', save_data_dir="/mocked/data/dir/fine-tune-pairs", system_prompt="pytest")

        # Sample conversations
        conversations = [
            [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}],
            [{'role': 'user', 'content': 'Goodbye'}, {'role': 'assistant', 'content': 'Bye'}]
        ]

        # Setup mock for json.dumps
        mock_json_dumps.side_effect = lambda x: f"mocked_json_{x}"

        # Call save_conversations
        fine_tuner.save_conversations(conversations)

        # Verify file creation
        mock_file_open.assert_called_once_with("/mocked/data/dir/fine-tune-pairs/conversations.jsonl", 'w', encoding='iso-8859-1')

        # Check the calls to json.dumps
        expected_json_calls = [call({"messages": conversation}) for conversation in conversations]
        mock_json_dumps.assert_has_calls(expected_json_calls, any_order=True)

        # Check the content written to the file
        handle = mock_file_open()
        expected_file_content = [f"mocked_json_{{'messages': {conversation}}}\n" for conversation in conversations]
        handle.write.assert_has_calls([call(content) for content in expected_file_content], any_order=True)

        # Assert that mock_file_open has been called twice
        assert mock_file_open.call_count == 2
