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

    complete_parser.add_argument('-a', '--analyzers', nargs='+', type=str, default=[], help='The list of analyzers to use')
    complete_parser.add_argument('-mf', '--model_first', type=str, default="ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d", help='The model to use for the first completion')
    complete_parser.add_argument('-ml', '--model_feedback', type=str, default="ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d", help='The model to use for the feedback loop')
    complete_parser.add_argument('-fv', '--feedback_variant', type=str, default="coworker", help='The feedback variant to use')
    complete_parser.add_argument('-p', '--system_prompt', type=str, default=None, help='The system prompt to use')
    complete_parser.add_argument('-i', '--max_iter', type=int, default=10, help='The maximum number of iterations to run the feedback loop')
    complete_parser.add_argument('-b', '--backtracking', action='store_true', help='Whether to use backtracking for the feedback loop')
    complete_parser.add_argument('-l', '--log_conversation', action='store_true', help='Whether to log the conversation')
    complete_parser.add_argument('-np', '--n_procs', type=int, default=1, help='The number of processes to use for the analyzers')
    complete_parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    complete_parser.add_argument('-t', '--temperature', type=float, default=0.7, help='The temperature to use for the first completion')
    complete_parser.add_argument('-mt', '--meta_temperature', type=float, default=0.0, help='The temperature to use for the feedback loop')
    complete_parser.add_argument('-uq', '--require_unique_sampling', action='store_true', help='Whether to require unique sampling for the feedback loop')

    complete_parser.add_argument('-n', '--n_completions', type=int, default=1, help='The number of completions to generate')
    complete_parser.add_argument('-f', '--file', type=str, help='File to complete')
    complete_parser.add_argument('-u', '--user_input', type=str, default='', help='User input to complete')

    # Add specific arguments to the evaluate command
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('-c', '--config', nargs='+', type=str, default=None, help='Configuration file in the configs folder for evaluation')
    evaluate_parser.add_argument('-n', '--n_repeat', type=int, default=1, help='Number of times to repeat the evaluation')

    metrics_parser = subparsers.add_parser('metrics')
    metrics_parser.add_argument('-c', '--config', nargs='+', type=str, default=None, help='Configuration file in the configs folder for evaluation')
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
            from llmcoder.llmcoder import LLMCoder

            if args.file is None and args.user_input == '':
                print('Either a file or user input must be provided, usage:')
                print('llmcoder complete -f <file> or llmcoder complete -u <user_input>')
                sys.exit(1)

            llmcoder = LLMCoder(
                analyzers=args.analyzers,
                model_first=args.model_first,
                model_feedback=args.model_feedback,
                feedback_variant=args.feedback_variant,
                system_prompt=args.system_prompt,
                max_iter=args.max_iter,
                log_conversation=args.log_conversation,
                n_procs=args.n_procs,
                verbose=args.verbose
            )

            if args.file:
                with open(args.file, 'r') as file:
                    user_input = file.read()
            else:
                user_input = args.user_input

            completion = llmcoder.complete(
                code=user_input,
                temperature=args.temperature,
                meta_temperature=args.meta_temperature,
                n=args.n_completions,
                require_unique_choices=args.require_unique_sampling)

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
