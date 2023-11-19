import os
import shutil
import subprocess
import tempfile

import requests
from tqdm import tqdm

from llmcoder.utils import get_data_dir

GITHUB_API = "https://api.github.com"


class GitHubScraper:
    """
    A class for scraping GitHub repositories and storing them in a flat structure.
    """
    def __init__(self, access_token: str | None = None, scraped_files_dir: str | None = None) -> None:
        """
        Initialize the GitHubScraper class with a GitHub access token.

        Parameters
        ----------
        access_token : str
            A GitHub access token for authenticating with the GitHub API.
        scraped_files_dir : str
            The directory to store the scraped files in, defaults to 'scraped_repos'.
        """
        self.access_token = access_token

        if scraped_files_dir is None:
            self.scraped_files_dir = get_data_dir("scraped_repos")  # /data/scraped_repos
        else:
            self.scraped_files_dir = scraped_files_dir

        os.makedirs(self.scraped_files_dir, exist_ok=True)

    def get_repos_with_query(self, query: str, num_repos: int = 1) -> list:
        """
        Fetch repositories using a specific GitHub API query.

        Parameters
        ----------
        query : str
            A GitHub API query.

        num_repos : int
            The number of repositories to fetch.
        """
        if self.access_token is not None:
            headers = {'Authorization': f'token {self.access_token}'}
        else:
            # Still works but may be rate limited
            headers = {}

        params: dict = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': num_repos
        }
        response = requests.get(f"{GITHUB_API}/search/repositories", headers=headers, params=params)

        if response.status_code == 200:
            return response.json()['items']
        else:
            print(f"Failed to fetch repositories: {response.status_code}")
            return []

    def get_popular_repos(self, num_repos_per_query: int = 1, queries: list[str] | None = None) -> list[str]:
        """
        Fetch popular repositories using different queries.

        Parameters
        ----------
        num_repos_per_query : int
            The number of repositories to fetch for each query.
        queries : list[str]
            A list of GitHub API queries to use for fetching repositories.

        Returns
        -------
        list[str]
            A list of repository URLs.
        """
        if queries is None:
            queries = [
                'language:python',
                'django in:name,description',
                'flask in:name,description',
                'data-science in:name,description',
                'machine-learning in:name,description'
            ]

        # Fetch repositories for each query
        repos = []
        for query in queries:
            repos.extend(self.get_repos_with_query(query, num_repos_per_query))

        # Remove potential duplicates
        unique_repos = {repo['id']: repo for repo in repos}.values()

        # Get the repository URLs with repo['html_url']
        return [repo['html_url'] for repo in unique_repos]

    def clone_repo(self, repo_url: str, output_dir: str) -> None:
        """
        Clone a repository into a specified directory by running a subprocess.

        Parameters
        ----------
        repo_url : str
            The URL of the repository to clone.
        output_dir : str
            The directory to clone the repository into.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        subprocess.run(["git", "clone", repo_url, output_dir])

    def extract_python_files(self, repo_dir: str, output_dir: str) -> None:
        """
        Extract all Python files from a repository and place them in a flat structure in the output directory.

        Parameters
        ----------
        repo_dir : str
            The directory of the repository to extract Python files from.
        output_dir : str
            The directory to place the extracted Python files in.
        """
        # Create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Walk through the repository directory and copy all Python files to the output directory
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    shutil.copy(file_path, os.path.join(output_dir, file))

    def accumulate_repositories(self, repository_sets: list[list[str]] | None = None) -> list[tuple[str, str]]:
        """
        Accumulate a list of repositories to scrape.

        Parameters
        ----------
        repository_sets : list[list[str]]
            A list of lists of repository URLs to scrape. Each list represents a set of repositories to scrape relating to a specific topic.

        Returns
        -------
        list[str]
            A list of repository URLs to scrape.
        """
        if repository_sets is None:
            # Get the top 10 Python repositories by stars
            all_popular_repos = self.get_popular_repos(num_repos_per_query=3)

            # Keep the top 10 repositories and put the rest (5) in a backup list
            popular_repos = all_popular_repos[:10]
            backup_repos = all_popular_repos[10:]

            # Get 5 of the most widely used Python packages
            popular_package_repos_urls = [
                "https://github.com/numpy/numpy",
                "https://github.com/pandas-dev/pandas",
                "https://github.com/matplotlib/matplotlib",
                "https://github.com/scikit-learn/scikit-learn",
                "https://github.com/python-pillow/Pillow"
            ]

            # Get some personal repositories
            personal_repos_urls = [
                "https://github.com/psaegert/pmtrendviz",
                "https://github.com/psaegert/nli-nec"
            ]

            # Get some repositories that I like
            liked_repos_urls = [
                "https://github.com/graphdeco-inria/gaussian-splatting",
                "https://github.com/lllyasviel/ControlNet",
                "https://github.com/maltfield/awesome-lemmy-instances",
                "https://github.com/Aleph-Alpha/aleph-alpha-client",
                "https://github.com/MaartenGr/BERTopic",
                "https://github.com/MilesCranmer/PySR",
                "https://github.com/AUTOMATIC1111/stable-diffusion-webui",
                "https://github.com/microsoft/Codex-CLI",
            ]

            # Combine the lists into a list of tuples of (repo_url, repo_name)
            repos_urls = popular_package_repos_urls + personal_repos_urls + liked_repos_urls
            repos_names = [url.split("/")[-1] for url in repos_urls]
            repos = list(zip(repos_urls, repos_names))

            # Add the popular repositories to the list
            repos.extend([(repo_url, repo_name) for repo_url in popular_repos for repo_name in repos_names])

            # Lastly, add the backup repositories (they will be skipped in case the goal of 25 is reached)
            repos.extend([(repo_url, repo_name) for repo_url in backup_repos for repo_name in repos_names])

        else:
            repos = []
            for repository_set in repository_sets:
                repos.extend([(repo_url, repo_name) for repo_url in repository_set for repo_name in repos_names])

        # Make sure there are no duplicates to avoid sampling issue later
        if not len(set(repos)) == len(repos):
            # Find the duplicate
            seen = set()
            for repo in repos:
                if repo in seen:
                    print(repo)
                else:
                    seen.add(repo)
            raise Exception("Duplicate repositories found")

        return repos

    def scrape_repositories(self, repos: list[tuple[str, str]] | None = None, max_n_repositories: int = 25, verbose: bool = False) -> None:
        """
        Scrape a list of repositories.

        Parameters
        ----------
        repos : list[tuple[str, str]]
            A list of tuples of (repo_url, repo_name).
        max_n_repositories : int
            The maximum number of repositories to scrape.
        verbose : bool
            Whether to show a progress bar.
        """
        pbar = tqdm(repos) if verbose else repos

        temp_clone_dir = tempfile.mkdtemp()

        for repo_url, repo_name in pbar:  # type: ignore
            # Clone the directolry into a temporary directory
            clone_repo_dir = os.path.join(temp_clone_dir, repo_name)

            # Then extract the Python files into the output directory
            output_repo_dir = os.path.join(self.scraped_files_dir, repo_name)

            # Check if the repository has already been cloned
            if os.path.exists(os.path.join(self.scraped_files_dir, repo_name)):
                continue

            if len(os.listdir(self.scraped_files_dir)) >= max_n_repositories:
                continue

            # Clone the repository and extract the Python files
            self.clone_repo(repo_url, clone_repo_dir)
            self.extract_python_files(clone_repo_dir, output_repo_dir)

            # If the repository is empty, remove the directory
            if len(os.listdir(output_repo_dir)) == 0:
                shutil.rmtree(output_repo_dir)
                continue

            # Add the repository URL to the 'repositories.txt' file to keep track of the source of the files
            with open(os.path.join(self.scraped_files_dir, "repositories.txt"), "a") as file:
                file.write(repo_url + "\n")

            # Remove the cloned repository directory
            shutil.rmtree(clone_repo_dir)

        # Remove the temporary directory
        shutil.rmtree(temp_clone_dir)
