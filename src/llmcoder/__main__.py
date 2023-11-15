import argparse
import sys


def main() -> None:
    """
    Parse the command line arguments for commands and options

    Commands:
    `llmcoder fine-tune-autocomplete` - Scrape & preprocess data for fine-tuning the autocomplete model
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='LLMcoder - Feedback-Based Coding Assistant')
    parser.add_argument('command', help='Command to execute')

    args = parser.parse_args()

    # Execute the command
    match args.command:
        case 'fine-tune-autocomplete':
            from llmcoder.fine_tune import fine_tune_autocomplete
            fine_tune_autocomplete()  # FIXME: Implement fine-tune-autocomplete
        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
