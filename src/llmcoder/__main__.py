import argparse
import sys


def main() -> None:
    """
    Parse the command line arguments for commands and options

    Commands:
    `llmcoder fine-tune-prepare` - Scrape & preprocess data for fine-tuning the autocomplete model
    `llmcoder fine-tune-export` - Export the preprocessed data for fine-tuning the autocomplete model
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='LLMcoder - Feedback-Based Coding Assistant', add_help=True)
    parser.add_argument('command', help='Command to execute. Available commands: fine-tune-preprocess, fine-tune-export')

    args = parser.parse_args()

    # Execute the command
    match args.command:
        case 'fine-tune-preprocess':
            from llmcoder.finetune import FineTunePreprocessor, GitHubScraper

            gh_scraper = GitHubScraper()
            repos = gh_scraper.accumulate_repositories()  # Use the default repos
            gh_scraper.scrape_repositories(repos=repos)  # Scrape the repos to the default directory

            preprocessor = FineTunePreprocessor()
            file_splits = preprocessor.preprocess()
            preprocessor.save_pairs(file_splits)

        case 'fine-tune-export':
            from llmcoder.finetune import FineTunePreprocessor

            preprocessor = FineTunePreprocessor()
            conversations = preprocessor.build_conversations()
            preprocessor.validate_conversations(conversations)
            preprocessor.save_conversations(conversations)

        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
