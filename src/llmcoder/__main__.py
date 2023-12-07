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
    preprocess_parser.add_argument('-n', '--name', type=str, help='Name of the dataset')

    export_parser = subparsers.add_parser('export')
    # Add specific arguments to the export command
    export_parser.add_argument('-n', '--name', type=str, help='Name of the dataset')

    # Add specific arguments to the complete command
    complete_parser = subparsers.add_parser('complete')
    complete_parser.add_argument('-f', '--file', type=str, help='File to complete')
    complete_parser.add_argument('-l', '--log', action='store_true', help='Log the conversation')
    complete_parser.add_argument('user_input', nargs='?', default='', help='User input to complete')

    # Add specific arguments to the evaluate command
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('-c', '--config', type=str, help='Configuration file in the configs folder for evaluation')
    # Add two more arguments to the evaluate command (--predict-only and --analysis-only, by default False)
    evaluate_parser.add_argument('-p', '--predict-only', action='store_true', help='Only run the prediction step')
    evaluate_parser.add_argument('-a', '--analysis-only', action='store_true', help='Only run the analysis step')
    # Add an additional argument to the evaluate command (--store-predictions, and --store-analysis, by default True)
    evaluate_parser.add_argument('--store-predictions', action='store_false', help='Store the predictions')
    evaluate_parser.add_argument('--store-analysis', action='store_false', help='Store the analysis')
    # Add an additional argument for the analysis step, given a stored results file
    evaluate_parser.add_argument('-f', '--results-file', type=str, help='File with the results of the prediction step')

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
            import os

            from dynaconf import Dynaconf

            from llmcoder.eval.evaluate import Evaluation
            from llmcoder.utils import get_config_dir

            config = Dynaconf(settings_files=[os.path.join(get_config_dir(), args.config)])

            eval = Evaluation(config)

            print('Running evaluation with the following configuration:')
            print(config)

            print('Storing predictions: ', args.store_predictions)
            print('Storing analysis: ', args.store_analysis)

            if args.predict_only and not args.analysis_only:
                print('Running prediction step')
                eval.predict(verbose=True, store_predictions=args.store_predictions)
            elif args.analysis_only and not args.predict_only:
                print('Running analysis step for results file: ', args.results_file)
                eval.analyze(args.results_file, verbose=True, store_analysis=args.store_analysis)
            else:
                print('Running evaluation step (prediction and analysis))')
                eval.run(verbose=True, store_analysis=args.store_analysis, store_predictions=args.store_predictions)

        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
