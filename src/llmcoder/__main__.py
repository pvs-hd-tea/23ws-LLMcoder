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
    preprocess_parser = subparsers.add_parser('preprocess')
    export_parser = subparsers.add_parser('export')
    complete_parser = subparsers.add_parser('complete')

    # Add specific arguments to the preprocess command
    preprocess_parser.add_argument('-n', '--name', type=str, help='Name of the dataset')
    export_parser.add_argument('-s', '--samples', type=str, help='Number of samples from each repository')

    # Add specific arguments to the export command
    export_parser.add_argument('-n', '--name', type=str, help='Name of the dataset')

    # Add specific arguments to the complete command
    complete_parser.add_argument('-f', '--file', type=str, help='File to complete')
    complete_parser.add_argument('-l', '--log', action='store_true', help='Log the conversation')
    complete_parser.add_argument('user_input', nargs='?', default='', help='User input to complete')

    # Parse the command line arguments
    args = parser.parse_args()

    # Execute the command
    match args.command:
        case 'preprocess':
            from llmcoder.data import FineTunePreprocessor, GitHubScraper

            gh_scraper = GitHubScraper(args.name)
            repos = gh_scraper.accumulate_repositories()  # Use the default repos
            gh_scraper.scrape_repositories(repos=repos)  # Scrape the repos to the default directory

            preprocessor = FineTunePreprocessor(args.name)
            split_files_contents = preprocessor.sample_files(n_samples=args.size)
            file_splits = preprocessor.preprocess(split_files_contents)
            preprocessor.save_pairs(file_splits)

        case 'export':
            from llmcoder.data import FineTunePreprocessor

            preprocessor = FineTunePreprocessor(args.name)
            conversations = preprocessor.build_conversations()
            preprocessor.validate_conversations(conversations)
            preprocessor.save_conversations(conversations)
        case 'complete':
            from llmcoder.LLMCoder import LLMCoder

            llmcoder = LLMCoder(log_conversation=args.log)

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
