"""
Version 1 of Tree of Completions: create_conversation_file only on best completion.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import openai
import tiktoken

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.treeofcompletions.PriorityQueue import Conversation, PriorityQueue
from llmcoder.utils import get_conversations_dir, get_openai_key, get_system_prompt, get_system_prompt_dir


class LLMCoder:
    """
    Initialize the LLMCoder

    Parameters
    ----------
    analyzers : list[str], optional
        The list of analyzers to use, by default []
    model_first : str, optional
        The model to use for the first completion, by default "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d"
    model_feedback : str, optional
        The model to use for the feedback loop, by default "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d"
    feedback_variant : str, optional
        The feedback variant to use, by default "separate"
    system_prompt : str, optional
        The system prompt to use, by default the one used for preprocessing and fine-tuning
    max_iter : int, optional
        The maximum number of iterations to run the feedback loop, by default 10
    log_conversation : bool, optional
        Whether to log the conversation, by default False
    n_procs : int, optional
        The number of processes to use for the analyzers, by default 1
    verbose : bool, optional
        Whether to print verbose output, by default True
    """
    def __init__(self,
                 analyzers: list[str] = None,
                 model_first: str = "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d",
                 model_feedback: str = "ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d",
                 feedback_variant: str = "coworker",
                 system_prompt: str | None = None,
                 max_iter: int = 10,
                 log_conversation: bool = True,
                 n_procs: int = 1,
                 verbose: bool = True) -> None:

        # Check for invalid feedback variants
        if feedback_variant not in ["separate", "coworker"]:
            raise ValueError("Inavlid feedback method")
        self.feedback_variant = feedback_variant

        # Remember the models
        self.model_first = model_first
        self.model_feedback = model_feedback

        # Set the feedback loop variables
        self.iterations = 0
        self.n_tokens_generated = 0
        self.encoder = tiktoken.get_encoding("p50k_base")
        self.max_iter = max_iter
        self.messages: list[dict[str, str]] = []

        # Set up the analyzers
        if analyzers is None:
            self.analyzers = {}
        else:
            self.analyzers = {
                analyzer: AnalyzerFactory.create_analyzer(analyzer, verbose=verbose) for analyzer in analyzers
            }

        # Set up the OpenAI API
        self.client = openai.OpenAI(api_key=get_openai_key())

        # Set up the number of processes to use for the analyzers
        self.n_procs = n_procs

        # Set up logging
        if log_conversation:
            self.conversation_file = self._create_conversation_file()
        else:
            self.conversation_file = None  # type: ignore

        # Get the system prompt
        if system_prompt is None:
            self.system_prompt = get_system_prompt()
        elif system_prompt in os.listdir(get_system_prompt_dir()):
            self.system_prompt = get_system_prompt(system_prompt)
        else:
            self.system_prompt = system_prompt

        self.verbose = verbose

        self._reset_loop()

    def _check_passing(self, conversation: Conversation) -> bool:
        """
        Check if all the analyzers passed in the last iteration of the conversation

        Parameters
        -------
        conversation: Conversation
            The conversation on which to be checked
        Returns
        -------
        bool
            True if all the analyzers passed, False otherwise
        """
        # If there was no iteration yet, return True
        if len(conversation.analyses) == 0:
            return True

        # Print how many critical analyzers have passed
        n_passed = sum(results['pass'] for results in conversation.analyses[-1].values() if (results['type'] == "critical" and type(results['pass']) is bool))
        n_total = len([results for results in conversation.analyses[-1].values() if results['type'] == "critical"])

        if self.verbose:
            print(f"[LLMcoder] {n_passed} / {n_total} analyzers passed")

        # If all the analyzers passed, return True
        if n_passed == n_total:
            return True

        # Otherwise, return False
        return False

    def _get_passing_conversations(self) -> list[Conversation]:
        """
        Get the conversations that passed all the analyzers

        Returns
        -------
        list[Conversation]
            The conversations that passed all the analyzers
        """
        return [conversation for conversation in self.conversations if self._check_passing(conversation)]

    def get_best_completion(self, conversations: list[Conversation]) -> str:
        return sorted(conversations, key=lambda conversation: conversation.score, reverse=True)[0].get_last_message()

    def complete(self, code: str, temperature: float = 0.7, n: int = 1) -> str:
        """
        Main entry point for LLMCoder.
        Complete the provided code with the LLMCoder feedback loop

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 1

        Returns
        -------
        str
            The completed code
        """
        # Reset the feedback loop and internal variables
        self._reset_loop()

        # Get the first completion with
        if self.verbose:
            print("[LLMcoder] Creating first completions...")
        self.step(code, temperature, n)

        # If the first completion is already correct, return it
        passing_conversations = self._get_passing_conversations()
        if len(passing_conversations) > 0:
            if self.verbose:
                print("[LLMcoder] First completion is correct. Stopping early...")
            return self.get_best_completion(passing_conversations)

        # Otherwise, start the feedback loop (but only if there are analyzers that can be used)
        if self.verbose:
            print("[LLMcoder] Starting feedback loop...")

        if len(self.analyzers) > 0 and self.max_iter > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for i in range(self.max_iter):
                if self.verbose:
                    print(f"[LLMcoder] Starting feedback iteration {i + 1}...")

                self.step(code, temperature, n)

                # If the code is correct, stop the feedback loop
                passing_conversations = self._get_passing_conversations()
                if len(passing_conversations) > 0:
                    if self.verbose:
                        print("[LLMcoder] Code is correct. Stopping early...")
                    return self.get_best_completion(passing_conversations)

        # No conversation passes, so just return the best completion
        return self.get_best_completion(self.conversations)

    @classmethod
    def _create_conversation_file(cls) -> str:
        """
        Create the conversation file storing the best conversation.

        Returns
        -------
        str
            The path to the conversation file
        """
        return os.path.join(get_conversations_dir(create=True), f"{datetime.now()}.jsonl")

    def _is_bad_completion(self, completion: str) -> bool:
        """
        Check if the completion already appeared in any of the existing conversations.
        If the assistant repeats a mistake, we do not want to consider it again.

        Parameters
        ----------
        completion : str
            The completion to check

        Returns
        -------
        bool
            True if the completion already appeared in any of the existing conversations, False otherwise
        """
        for conversation in self.conversations:
            for message in conversation.messages:
                if message["content"] == completion:
                    return True

        return False

    def _get_completions_for(self, conversation: Conversation, model: str = 'gpt-3.5-turbo', temperature: float = 0.7, n: int = 1, max_retries: int = 10) -> None:
        """
        Use OpenAI's API to get completion(s) for the user's code for a given conversation

        Parameters
        ----------
        conversation: Conversation
            Tuple in the priority queue. Contains the completion/code over which the model will complete.
        model : str, optional
            The model to use for the completion, by default 'gpt-3.5-turbo'
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 1
        max_retries : int, optional
            The maximum number of retries to get a valid completion, by default 10
        """
        # Get the completions from OpenAI's API -> 3 completions stored in candidates.choices
        candidates = self.client.chat.completions.create(
            messages=conversation.messages,
            model=model,
            temperature=temperature,
            n=n)  # type: ignore

        # Count the number of tokens generated
        self.n_tokens_generated += sum([len(self.encoder.encode(message.message.content)) for message in candidates.choices])

        # Filter out completions that are repetitions of previous mistakes
        valid_choices = [completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]

        # If all completions are repetitions of previous mistakes, increase the temperature and the number of choices until we get a valid completion
        increased_temperature = temperature
        increased_n = n
        repetition = 0

        while len(valid_choices) < n and repetition < max_retries:
            if self.verbose:
                print(f"[LLMcoder] All completions are repetitions of previous mistakes. Increasing temperature to {increased_temperature} and number of choices to {increased_n}... [repetition {repetition + 1}/{max_retries}]")

            # Sample new candidates
            candidates = self.client.chat.completions.create(
                messages=conversation.messages,
                model=model,
                temperature=increased_temperature,
                n=increased_n)  # type: ignore

            # Filter out completions that are repetitions of previous mistakes
            valid_choices = [completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]

            increased_temperature = min(2, increased_temperature + 0.1)
            increased_n = min(32, increased_n * 2)
            repetition += 1

        # If we still do not have valid choices, abort
        if repetition >= max_retries:
            if self.verbose:
                print("[LLMcoder] All completions are repetitions of previous mistakes. Aborting...")
            return None

        # Now that we have valid choices, run the analyzers on them in parallel and determine the best one
        if n > 1 and len(valid_choices) > 1:
            if self.verbose:
                print(f"[LLMcoder] Analyzing {len(candidates.choices)} completions...")

            with ThreadPoolExecutor(max_workers=self.n_procs) as executor:
                # Create a mapping of future to completion choice
                choice_to_future = {
                    i: executor.submit(self.run_analyzers, conversation.messages[1]["content"], choice.message.content)
                    for i, choice in enumerate(valid_choices)}

                # Retrieve results in the order of valid_choices
                for i in range(len(valid_choices)):
                    future = choice_to_future[i]
                    try:
                        # Update the analyzer results history with the results of total completion
                        analysis = future.result()
                        analysis_score = [sum([results["score"] for results in analysis.values()])]

                        if self.verbose:
                            print(f"[LLMcoder] Added completion {i} with score: {analysis_score}")
                        self.conversations.push(conversation.copy().add_analysis(analysis).add_message(valid_choices[i].message.content))
                    except Exception as e:
                        if self.verbose:
                            print(f"[LLMcoder] An exception occurred during analysis of choice: {e}")
                        raise e

        else:
            # If we only have one completion, still run the analyzers on it
            if self.verbose:
                print("[LLMcoder] Analyzing completion...")
            analysis = self.run_analyzers(conversation.messages[1]["content"], valid_choices[0].message.content)

            # There is only one valid choice
            # Update the analyzer results history with the results of the completion
            self.conversations.push(conversation.copy().add_analysis(analysis).add_message(valid_choices[0].message.content))

    def _add_message_to_conversation(self, role: str, conversation: Conversation, message: str | None = None, model: str = 'gpt-3.5-turbo', temperature: float = 0.7, n: int = 1) -> bool:
        """
        Add a message to a scpecific conversation in PQ.

        Parameters
        ----------
        conversation: Conversation
            Tuple in the priority queue. A message will be added in continuity with the string completion.
        role : str
            The role of the message
        message : str, optional
            The message to add, by default None
        model : str, optional
            The model to use for the completion, by default 'gpt-3.5-turbo'
        temperature : float, optional
            The temperature to use for the assistant completion, by default 0.7
        n : int, optional
            The number of assistant choices to generate, by default 1

        Returns
        -------
        bool
            True if the message was added, False otherwise
        """
        # If the user is the assistant, generate a response
        if role == "assistant" and message is None:
            self._get_completions_for(conversation, model, temperature, n)
        else:
            # Add the message to the conversation
            success = conversation.add_message({"role": role, "content": message})

        if self.verbose:
            print(f"[LLMcoder] --- {role.upper()} --- :\n{message} ----------------------")

        # If the conversation should be logged, log it
        if self.conversation_file is not None:
            # If the conversation file already exists, only append the last message as a single line
            if os.path.isfile(self.conversation_file):
                with open(self.conversation_file, "a") as f:
                    f.write(json.dumps(conversation.messages[1], ensure_ascii=False) + "\n")
            # Otherwise, write the whole conversation
            else:
                with open(self.conversation_file, "w") as f:
                    for message in conversation.messages:
                        f.write(json.dumps(message, ensure_ascii=False) + "\n")

        # The message was added successfully
        return success

    def _reset_loop(self) -> None:
        """
        Reset the feedback loop and internal variables. Also re-add the system prompt as the first message
        """
        # Reset the feedback loop variables
        self.iterations = 0
        self.n_tokens_generated = 0

        # Create a tree of completions: initialize values to default
        self.conversations = PriorityQueue(
            Conversation(
                score=0,  # Score does not matter here because we pop the conversation with the highest score anyway
                messages=[{
                    "role": "system",
                    "content": self.system_prompt}],
                analyses=[])
        )

    def run_analyzers(self, code: str, completion: str) -> dict[str, dict]:
        """
        Run the analyzers on the code and completion

        Parameters
        ----------
        code : str
            The code to analyze
        completion : str
            The completion to analyze

        Returns
        -------
        dict[str, dict]
            The analyzer results
        """
        analyzer_results: dict[str, dict] = {}

        # In separete mode, each analyzer is run separately without a shared context
        if self.feedback_variant == "separate":
            if self.verbose:
                print("[LLMcoder] Analyzing code in separate mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                if self.verbose:
                    print(f"[LLMcoder] Running {analyzer_name}...")
                analyzer_results[analyzer_name] = analyzer_instance.analyze(code, completion)

        # In coworker mode, the analyzers are run in parallel and share a context
        elif self.feedback_variant == "coworker":
            if self.verbose:
                print("[LLMcoder] Analyzing code in coworker mode...")
            for analyzer_name, analyzer_instance in self.analyzers.items():
                if self.verbose:
                    print(f"[LLMcoder] Running {analyzer_name}...")
                analyzer_results[analyzer_name] = analyzer_instance.analyze(code, completion, context=analyzer_results)
                print(f"After analyzing, the results were {analyzer_results[analyzer_name].values()}")

        # Return the collected analyzer results
        return analyzer_results

    def _feedback_prompt_template(self, result_messages: list[str]) -> str:
        """
        Create the feedback prompt template for the analyzer results

        Parameters
        ----------
        result_messages : list[str]
            The analyzer result messages

        Returns
        -------
        str
            The feedback prompt template
        """
        return '[INST]\n' + '\n'.join(result_messages) + '\n\nFix, improve and rewrite your completion for the following code:\n[/INST]\n'

    def step(self, code: str, temperature: float = 0.7, n: int = 1) -> None:
        """
        Complete the provided code with the OpenAI model and feedback, if available
        Make choice on highest scored snippet through PriorityQueue.pop().

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        n : int, optional
            The number of choices to generate, by default 1
        """
        # Choose highest-scored conversation from the priority queue
        most_promising_conversation = self.conversations.pop()

        # If there is feedback available from previous analyses, add it to the prompt
        if len(most_promising_conversation.analyses) > 0:
            feedback_prompt = self._feedback_prompt_template(
                [str(result['message']) for result in most_promising_conversation.analyses[-1].values()
                 if (not result['pass'] or result['type'] == "info")])

        # If there is not feedback available, the prompt will just be the user's code
        else:
            print("[LLMcoder] There are no analyzer results")
            feedback_prompt = ""

        # Add the prompt to the messages
        self._add_message_to_conversation("user", most_promising_conversation, message=feedback_prompt + code)

        # Get n completions from the assistant
        # Clone the parent conversation n times and append the completions
        # Then, return whether the completion generation was successful
        self._add_message_to_conversation(
            "assistant",
            most_promising_conversation,
            model=self.model_feedback,
            temperature=temperature,
            n=n)

        self.iterations += 1
