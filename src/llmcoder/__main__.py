# AI ToolsFirst prototype for completion fetching documentation
import argparse
import sys


def main() -> None:
    """
    Parse the command line arguments for commands and options

    Commands:
    `llmcoder fine-tune-preprocess`: Preprocess the data for fine-tuning
    `llmcoder fine-tune-export`: Export the fine-tuned model
    `llmcoder complete`: Complete a piece of code
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='LLMcoder - Feedback-Based Coding Assistant')
    parser.add_argument('command', help='Command to execute')

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
        case 'complete':
            # TODO: Add option to specify the analyzers to use
            from llmcoder.LLMCoder import LLMCoder

            # Creating an instance of LLMCoder
            llmcoder = LLMCoder()
            user_input = "def say_something_nice():\n"

            completion = llmcoder.complete(user_input)
            print(completion)
        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
