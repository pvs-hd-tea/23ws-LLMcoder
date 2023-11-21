# AI ToolsFirst prototype for completion fetching documentation
import argparse
import sys
from .LlmCoder import LLMCoder



def main() -> None:
    """
    Parse the command line arguments for commands and options

    Commands:
    `llmcoder fine-tune-autocomplete` - Scrape & preprocess data for fine-tuning the autocomplete model
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description = 'LLMcoder - Feedback-Based Coding Assistant')
    parser.add_argument('command', help = 'Command to execute')

    args = parser.parse_args()

    # Execute the command
    # match args.command:
    if(args.command =='fine-tune-autocomplete'):
        # from llmcoder.fine_tune import fine_tune_autocomplete
        # fine_tune_autocomplete()  # FIXME: Implement fine-tune-autocomplete
        # Creating an instance of LLMCoder
        llm_coder_instance = LLMCoder(model_name = "gpt-3.5-turbo", feedback_variant = "separate", analyzers_list = {})
        user_input = "hello"
        result = llm_coder_instance.complete_first(user_input)
        print(result)

    else:
        print('Unknown command: ', args.command)
        sys.exit(1)
