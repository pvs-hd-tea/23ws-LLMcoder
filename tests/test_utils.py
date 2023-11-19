import os
import shutil
from unittest.mock import Mock, mock_open, patch

import pytest

from llmcoder.utils import get_data_dir, get_github_access_token, get_openai_key


# Test get_data_dir function
def test_get_data_dir() -> None:
    data_dir = get_data_dir()

    assert os.path.exists(data_dir)
    assert os.path.abspath(data_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def test_get_data_dir_argument() -> None:
    data_dir = get_data_dir("pytest")

    assert os.path.exists(data_dir)
    assert os.path.abspath(data_dir) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'pytest'))

    # Remove the pytest directory
    shutil.rmtree(data_dir)


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
