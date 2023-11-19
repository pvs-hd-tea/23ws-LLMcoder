import os


def get_data_dir(*args: str) -> str:
    """
    Get the path to the data directory.

    Parameters
    ----------
    args : str
        The path to the data directory.

    Returns
    -------
    str
        The path to the data directory.
    """
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'data', *args), exist_ok=True)

    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', *args)


def get_openai_key(key: str) -> str:
    """
    Get OpenAI API key. Try to interpret the key as a key first, then as a path to a file containing the key.
    Finally, fall back to the default key.txt file or the OPENAI_KEY environment variable.

    Parameters
    ----------
    key : str
        The key or path to the key file.

    Returns
    -------
    str
        The key.
    """

    # Try to interpret the key as a key first (they start with "sk-")
    if key.startswith("sk-"):
        return key

    # Try to interpret the key as a path to a file containing the key
    if os.path.isfile(key):
        with open(key, "r") as f:
            return f.read().strip()

    # Fall back to the default key.txt file
    if os.path.isfile("key.txt"):
        with open(os.path.join(__file__, '..', '..', 'key.txt'), "r") as f:
            return f.read().strip()

    # Fall back to the OPENAI_KEY environment variable
    if "OPENAI_KEY" in os.environ:
        return os.environ["OPENAI_KEY"]

    raise ValueError("Could not find OpenAI API key. Please provide it as an argument or in a key.txt file.")


def get_github_access_token(token: str) -> str:
    """
    Get GitHub access token. Try to interpret the token as a token first, then as a path to a file containing the token.
    Finally, fall back to the default token.txt file or the GITHUB_ACCESS_TOKEN environment variable.

    Parameters
    ----------
    token : str
        The token or path to the token file.

    Returns
    -------
    str
        The token.
    """

    # Try to interpret the token as a token first (they start with "github_pat")
    if token.startswith("github_pat"):
        return token

    # Try to interpret the token as a path to a file containing the token
    if os.path.isfile(token):
        with open(token, "r") as f:
            return f.read().strip()

    # Fall back to the default token.txt file
    if os.path.isfile("gh_access.txt"):
        with open(os.path.join(__file__, '..', '..', 'gh_access.txt'), "r") as f:
            return f.read().strip()

    # Fall back to the GITHUB_ACCESS_TOKEN environment variable
    if "GITHUB_ACCESS_TOKEN" in os.environ:
        return os.environ["GITHUB_ACCESS_TOKEN"]

    raise ValueError("Could not find GitHub access token. Please provide it as an argument or in a token.txt file.")
