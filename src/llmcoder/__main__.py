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

    # Add specific arguments to the preprocess command
    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('-n', '--name', required=True, type=str, help='Name of the dataset')
    preprocess_parser.add_argument('-s', '--size', type=int, help='Number of samples to extract per repository')

    export_parser = subparsers.add_parser('export')
    # Add specific arguments to the export command
    export_parser.add_argument('-n', '--name', required=True, type=str, help='Name of the dataset')

    # Add specific arguments to the complete command
    complete_parser = subparsers.add_parser('complete')
    complete_parser.add_argument('-f', '--file', type=str, help='File to complete')
    complete_parser.add_argument('-l', '--log', action='store_true', help='Log the conversation')
    complete_parser.add_argument('user_input', nargs='?', default='', help='User input to complete')

    # Add specific arguments to the evaluate command
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('-c', '--config', type=str, default=None, help='Configuration file in the configs folder for evaluation')
    evaluate_parser.add_argument('-n', '--n_repeat', type=int, default=1, help='Number of times to repeat the evaluation')

    metrics_parser = subparsers.add_parser('metrics')
    metrics_parser.add_argument('-c', '--config', type=str, default=None, help='Configuration file in the configs folder for evaluation')
    metrics_parser.add_argument('-i', '--index', type=int, default=None, help='Only compute metrics of the i-th repetition')

    # Parse the command line arguments
    args = parser.parse_args()

    # Execute the command
    match args.command:
        case 'preprocess':
            from llmcoder.data import GitHubScraper, Preprocessor

            gh_scraper = GitHubScraper(args.name)
            repos = gh_scraper.accumulate_repositories()  # Use the default repos
            gh_scraper.scrape_repositories(repos=repos)  # Scrape the repos to the default directory

            preprocessor = Preprocessor(args.name)
            split_files_contents = preprocessor.sample_files(n_samples=args.size)
            file_splits = preprocessor.preprocess(split_files_contents)
            preprocessor.save_pairs(file_splits)

        case 'export':
            from llmcoder.data import Preprocessor

            preprocessor = Preprocessor(args.name)
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
        case 'evaluate':
            from llmcoder.eval.evaluate import Evaluation
            _ = Evaluation(args.config).run(store=True, n_repeat=args.n_repeat, verbose=True)

        case 'metrics':
            from llmcoder.eval.evaluate import Metrics
            _ = Metrics(args.config).run(store=True, index=args.index, verbose=True)

        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
