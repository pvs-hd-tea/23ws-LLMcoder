# AI ToolsFirst prototype for completion fetching documentation
import argparse
import sys


def main() -> None:
    """
    Parse the command line arguments for commands and options
    """

    # Create the top-level parser
    parser = argparse.ArgumentParser(description='LLMcoder - Feedback-Based Coding Assistant')
    subparsers = parser.add_subparsers(dest='command')

    # Create subparser for each command
    # fine_tune_preprocess_parser = subparsers.add_parser('fine-tune-preprocess')
    # fine_tune_export_parser = subparsers.add_parser('fine-tune-export')
    complete_parser = subparsers.add_parser('complete')

    # Add specific arguments to the complete command
    complete_parser.add_argument('-f', '--file', type=str, help='File to complete')
    complete_parser.add_argument('user_input', nargs='?', default='', help='User input to complete')

    # Parse the command line arguments
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
            from llmcoder.LLMCoder import LLMCoder

            llmcoder = LLMCoder()

            if args.file:
                with open(args.file, 'r') as file:
                    user_input = file.read()
            else:
                user_input = args.user_input

            completion = llmcoder.complete(user_input)

            if args.file:
                with open(args.file, 'a') as file:
                    file.write(completion)
            else:
                print(completion)
        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
