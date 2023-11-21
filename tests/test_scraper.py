import os
import unittest
from unittest.mock import Mock, patch

from llmcoder.finetune import GitHubScraper
from llmcoder.finetune.scraper import GITHUB_API


# Generated with GPT-4 under supervision
class TestGitHubScraper(unittest.TestCase):
    def test_init_default_params(self) -> None:
        """
        Test the __init__ method with default parameters.
        """
        scraper = GitHubScraper()

        self.assertEqual(scraper.access_token, None)

        # Check if the default data directory exists
        self.assertTrue(os.path.exists(scraper.scraped_files_dir))

    @patch('requests.get')
    def test_get_repos_with_query_success(self, mock_get: Mock) -> None:
        """
        Test `get_repos_with_query` method for a successful API response.
        """
        # Setup
        mock_response = Mock()
        expected_data = [{'id': 12345, 'name': 'test_repo'}]
        mock_response.json.return_value = {'items': expected_data}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        scraper = GitHubScraper(access_token='test_token')
        query = 'test_query'
        num_repos = 1

        # Exercise
        result = scraper.get_repos_with_query(query, num_repos)

        # Verify
        mock_get.assert_called_once_with(
            f"{GITHUB_API}/search/repositories",
            headers={'Authorization': 'token test_token'},
            params={'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': num_repos}
        )
        self.assertEqual(result, expected_data)

    @patch('requests.get')
    def test_get_repos_with_query_failure(self, mock_get: Mock) -> None:
        """
        Test `get_repos_with_query` method for a failed API response.
        """
        # Setup
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        scraper = GitHubScraper()
        query = 'test_query'
        num_repos = 1

        # Exercise
        result = scraper.get_repos_with_query(query, num_repos)

        # Verify
        self.assertEqual(result, [])

    @patch('requests.get')
    def test_get_repos_with_query_no_token(self, mock_get: Mock) -> None:
        """
        Test `get_repos_with_query` method without an access token.
        """
        # Setup
        mock_response = Mock()
        expected_data = [{'id': 67890, 'name': 'test_repo_no_token'}]
        mock_response.json.return_value = {'items': expected_data}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        scraper = GitHubScraper()
        query = 'test_query'
        num_repos = 1

        # Exercise
        result = scraper.get_repos_with_query(query, num_repos)

        # Verify
        mock_get.assert_called_once_with(
            f"{GITHUB_API}/search/repositories",
            headers={},
            params={'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': num_repos}
        )
        self.assertEqual(result, expected_data)

    @patch.object(GitHubScraper, 'get_repos_with_query')
    def test_get_popular_repos_default_queries(self, mock_get_repos_with_query: Mock) -> None:
        """
        Test `get_popular_repos` method with default queries.
        """
        # Setup
        mock_get_repos_with_query.return_value = [
            {'id': 1, 'html_url': 'https://github.com/repo1'},
            {'id': 2, 'html_url': 'https://github.com/repo2'},
            {'id': 3, 'html_url': 'https://github.com/repo3'},
            {'id': 4, 'html_url': 'https://github.com/repo4'},
            {'id': 5, 'html_url': 'https://github.com/repo5'}
        ]
        scraper = GitHubScraper()

        # Default queries
        default_queries = [
            'language:python',
            'django in:name,description',
            'flask in:name,description',
            'data-science in:name,description',
            'machine-learning in:name,description'
        ]

        # Exercise
        result = scraper.get_popular_repos()

        # Verify
        self.assertEqual(len(result), len(default_queries) * 1)  # Each query returns 1 repos
        for query in default_queries:
            mock_get_repos_with_query.assert_any_call(query, 1)

    @patch.object(GitHubScraper, 'get_repos_with_query')
    def test_get_popular_repos_custom_queries(self, mock_get_repos_with_query: Mock) -> None:
        """
        Test `get_popular_repos` method with custom queries.
        """
        # Setup
        custom_queries = ['test_query1', 'test_query2']
        mock_get_repos_with_query.return_value = [{'id': 3, 'html_url': 'https://github.com/repo1'}, {'id': 4, 'html_url': 'https://github.com/repo2'}]
        scraper = GitHubScraper()

        # Exercise
        result = scraper.get_popular_repos(num_repos_per_query=1, queries=custom_queries)

        # Verify
        self.assertEqual(len(result), len(custom_queries) * 1)  # Each query returns 1 repo
        for query in custom_queries:
            mock_get_repos_with_query.assert_any_call(query, 1)

    @patch.object(GitHubScraper, 'get_repos_with_query')
    def test_get_popular_repos_duplicate_removal(self, mock_get_repos_with_query: Mock) -> None:
        """
        Test `get_popular_repos` method for removing duplicates.
        """
        # Setup
        duplicate_repos = [
            {'id': 1, 'html_url': 'https://github.com/repo1'},
            {'id': 1, 'html_url': 'https://github.com/repo1'}
        ]
        mock_get_repos_with_query.return_value = duplicate_repos
        scraper = GitHubScraper()

        # Exercise
        result = scraper.get_popular_repos()

        # Verify
        self.assertEqual(len(result), 1)
        self.assertEqual(result, ['https://github.com/repo1'])

    @patch('subprocess.run')
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_clone_repo(self, mock_exists: Mock, mock_makedirs: Mock, mock_run: Mock) -> None:
        """
        Test the `clone_repo` method.
        """
        repo_url = "https://github.com/example/repo.git"
        output_dir = "/path/to/output"

        # Setup
        mock_exists.return_value = False

        scraper = GitHubScraper()

        # Exercise
        scraper.clone_repo(repo_url, output_dir)

        # Verify
        mock_run.assert_called_once_with(["git", "clone", repo_url, output_dir])

    @patch('shutil.copy')
    @patch('os.walk')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_extract_python_files(self, mock_makedirs: Mock, mock_exists: Mock, mock_walk: Mock, mock_copy: Mock) -> None:
        """
        Test the `extract_python_files` method.
        """
        repo_dir = "/path/to/repo"
        output_dir = "/path/to/output"
        python_files = [("dir1", "file1.py"), ("dir2", "file2.py")]

        # Setup
        mock_exists.return_value = False
        # Adjusted to reflect the actual structure returned by os.walk
        mock_walk.return_value = [(os.path.join(repo_dir, root), [], [file]) for root, file in python_files]

        scraper = GitHubScraper()

        # Exercise
        scraper.extract_python_files(repo_dir, output_dir)

        # Verify
        for root, file in python_files:
            mock_copy.assert_any_call(os.path.join(repo_dir, root, file), os.path.join(output_dir, file))

    @patch.object(GitHubScraper, 'get_popular_repos')
    def test_accumulate_repositories_default(self, mock_get_popular_repos: Mock) -> None:
        """
        Test `accumulate_repositories` with default parameters.
        """
        # Setup
        mock_get_popular_repos.return_value = ['https://github.com/popular/repo1', 'https://github.com/popular/repo2']
        scraper = GitHubScraper()

        # Exercise
        result = scraper.accumulate_repositories()

        # Verify
        mock_get_popular_repos.assert_called_once_with(num_repos_per_query=3)
        self.assertTrue(isinstance(result, list))
        # Check that popular repositories are included in the result
        self.assertIn(('https://github.com/popular/repo1', 'repo1'), result)
        self.assertIn(('https://github.com/popular/repo2', 'repo2'), result)
        # Verify no duplicates
        self.assertEqual(len(result), len(set(result)))

    def test_accumulate_repositories_custom_sets(self) -> None:
        """
        Test `accumulate_repositories` with custom repository sets.
        """
        scraper = GitHubScraper()
        custom_sets = [
            ['https://github.com/custom/repo1', 'https://github.com/custom/repo2'],
            ['https://github.com/another/repo1']
        ]

        # Exercise
        result = scraper.accumulate_repositories(repository_sets=custom_sets)

        # Verify
        expected_repos = [url for sublist in custom_sets for url in sublist]
        expected_names = [url.split('/')[-1] for url in expected_repos]
        expected_list = list(zip(expected_repos, expected_names))

        self.assertEqual(result, expected_list)

    def test_accumulate_repositories_duplicates(self) -> None:
        """
        Test `accumulate_repositories` for handling duplicates.
        """
        scraper = GitHubScraper()
        duplicate_sets = [
            ['https://github.com/duplicate/repo', 'https://github.com/unique/repo'],
            ['https://github.com/duplicate/repo']
        ]

        # Exercise & Verify
        with self.assertRaises(Exception):
            scraper.accumulate_repositories(repository_sets=duplicate_sets)
