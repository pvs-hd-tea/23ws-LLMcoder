import os
import shutil
from unittest.mock import Mock, mock_open, patch

import pytest

from llmcoder.utils import get_config_dir, get_conversations_dir, get_data_dir, get_github_access_token, get_openai_key, get_system_prompt, get_system_prompt_dir


# Test get_data_dir function
def test_get_data_dir() -> None:
    data_dir = get_data_dir(create=True)

    assert os.path.exists(data_dir)
    assert os.path.abspath(data_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def test_get_data_dir_argument() -> None:
    data_dir = get_data_dir("pytest", create=True)

    assert os.path.exists(data_dir)
    assert os.path.abspath(data_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'pytest'))

    # Remove the pytest directory
    shutil.rmtree(data_dir)


# Test get_config_dir function
def test_get_config_dir() -> None:
    config_dir = get_config_dir(create=True)

    assert os.path.exists(config_dir)
    assert os.path.abspath(config_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))


def test_get_config_dir_argument() -> None:
    config_dir = get_config_dir("pytest", create=True)

    assert os.path.exists(config_dir)
    assert os.path.abspath(config_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'pytest'))

    # Remove the pytest directory
    shutil.rmtree(config_dir)


# Test get_system_prompt_dir function
def test_get_system_prompt_dir() -> None:
    system_prompt_dir = get_system_prompt_dir(create=True)

    assert os.path.exists(system_prompt_dir)
    assert os.path.abspath(system_prompt_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'system_prompts'))


def test_get_system_prompt_dir_argument() -> None:
    system_prompt_dir = get_system_prompt_dir("pytest", create=True)

    assert os.path.exists(system_prompt_dir)
    assert os.path.abspath(system_prompt_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'system_prompts', 'pytest'))

    # Remove the pytest directory
    shutil.rmtree(system_prompt_dir)


# Test get_openai_key function
def test_openai_direct_key() -> None:
    assert get_openai_key("sk-test_key") == "sk-test_key"


@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="sk-file_key")
def test_openai_valid_file_path(mock_file: Mock, mock_isfile: Mock) -> None:
    assert get_openai_key("path/to/key.txt") == "sk-file_key"


@patch("os.path.isfile", return_value=False)
def test_openai_invalid_file_path(mock_isfile: Mock) -> None:
    with pytest.raises(ValueError):
        get_openai_key("invalid/path/to/key.txt")


def test_openai_default_key_file() -> None:
    key_file_path = os.path.join(os.path.dirname(__file__), '..', 'key.txt')

    # Check if key.txt exists
    if os.path.isfile(key_file_path):
        with open(key_file_path, "r") as file:
            content = file.read().strip()
            assert content.startswith("sk-")
    else:
        # Create key.txt for the test
        test_key = "sk-test_default_key"
        with open(key_file_path, "w") as file:
            file.write(test_key)

        try:
            # Test get_openai_key function
            assert get_openai_key("") == test_key
        finally:
            # Cleanup: Remove the test key.txt file
            os.remove(key_file_path)


@patch.dict(os.environ, {"OPENAI_KEY": "sk-env_key"})
def test_openai_environment_variable() -> None:
    assert get_openai_key() == "sk-env_key"


@patch("os.path.isfile", return_value=False)
@patch.dict(os.environ, {}, clear=True)
def test_openai_key_not_found(mock_isfile: Mock) -> None:
    with pytest.raises(ValueError):
        get_openai_key()


# Test get_github_access_token function
def test_github_direct_token() -> None:
    assert get_github_access_token("github_pat_test_token") == "github_pat_test_token"


@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="file_token")
def test_github_valid_token_file(mock_file: Mock, mock_isfile: Mock) -> None:
    assert get_github_access_token("path/to/token.txt") == "file_token"


@patch("os.path.isfile", return_value=False)
def test_github_invalid_token_file(mock_isfile: Mock) -> None:
    with pytest.raises(ValueError):
        get_github_access_token("invalid/path/to/token.txt")


def test_github_default_token_file() -> None:
    token_file_path = os.path.join(os.path.dirname(__file__), '..', 'gh_access.txt')

    # Check if token.txt exists
    if os.path.isfile(token_file_path):
        with open(token_file_path, "r") as file:
            content = file.read().strip()
            assert content != ""
    else:
        # Create token.txt for the test
        test_token = "test_default_token"
        with open(token_file_path, "w") as file:
            file.write(test_token)

        try:
            # Test get_github_access_token function
            assert get_github_access_token("") == test_token
        finally:
            # Cleanup: Remove the test token.txt file
            os.remove(token_file_path)


@patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "github_pat_test_token"})
def test_github_environment_variable() -> None:
    assert get_github_access_token() == "github_pat_test_token"


@patch("os.path.isfile", return_value=False)
@patch.dict(os.environ, {}, clear=True)
def test_github_key_not_found(mock_isfile: Mock) -> None:
    with pytest.raises(ValueError):
        get_github_access_token()


# get_system_prompt
def test_get_system_prompt_file_exists() -> None:
    prompt_file_path = os.path.join(os.path.dirname(__file__), '..', 'system_prompts', '2023-11-15_GPT-Builder.txt')

    # Check if system_prompt.txt exists
    if os.path.isfile(prompt_file_path):
        with open(prompt_file_path, "r") as file:
            content = file.read().strip()
            assert content != ""
    else:
        # Create system_prompt.txt for the test
        test_prompt = "pytest_system_prompt"
        with open(prompt_file_path, "w") as file:
            file.write(test_prompt)

        try:
            # Test get_system_prompt function
            assert get_system_prompt() == test_prompt
        finally:
            # Cleanup: Remove the test system_prompt.txt file
            os.remove(prompt_file_path)


def test_get_system_prompt_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        get_system_prompt("invalid_prompt.txt")


# get_conversations_dir
def test_get_conversations_dir() -> None:
    conversations_dir = get_conversations_dir(create=True)

    assert os.path.exists(conversations_dir)
    assert os.path.abspath(conversations_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'conversations'))
