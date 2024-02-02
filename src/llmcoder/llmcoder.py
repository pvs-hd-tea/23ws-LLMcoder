import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import openai
import tiktoken

from llmcoder.analyze.factory import AnalyzerFactory
from llmcoder.conversation.conversation import Conversation
from llmcoder.conversation.priority_queue import PriorityQueue
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

    def _get_passing_conversations(self) -> list[Conversation]:
        """
        Get the conversations that passed all the analyzers

        Returns
        -------
        list[Conversation]
            The conversations that passed all the analyzers
        """
        return [c for c in self.conversations if c.passing]

    def _get_best_completion(self, conversations: list[Conversation]) -> str:
        """
        Get the best completion from the provided conversations

        Parameters
        ----------
        conversations : list[Conversation]
            The conversations to get the best completion from

        Returns
        -------
        str
            The best completion
        """
        return sorted(conversations, key=lambda c: c.score, reverse=True)[0].get_last_message()

    def complete(self, code: str, temperature: float = 0.7, meta_temperature: float = 0.0, n: int = 1) -> str:
        """
        Main entry point for LLMCoder.
        Complete the provided code with the LLMCoder feedback loop

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        meta_temperature : float, optional
            The temperature to use for choosing the most promising conversation, by default 0.1
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
        self._step(code=code, temperature=temperature, meta_temperature=meta_temperature, n=n)

        # If the first completion is already correct, return it
        passing_conversations = self._get_passing_conversations()
        if len(passing_conversations) > 0:
            if self.verbose:
                print("[LLMcoder] First completion is correct. Stopping early...")
            return self._get_best_completion(passing_conversations)

        # Otherwise, start the feedback loop (but only if there are analyzers that can be used)
        if self.verbose:
            print("[LLMcoder] Starting feedback loop...")

        if len(self.analyzers) > 0 and self.max_iter > 0:
            # Run the feedback loop until the code is correct or the max_iter is reached
            for i in range(self.max_iter):
                if self.verbose:
                    print(f"[LLMcoder] Starting feedback iteration {i + 1}...")

                self._step(code=code, temperature=temperature, meta_temperature=meta_temperature, n=n)

                # If the code is correct, stop the feedback loop
                passing_conversations = self._get_passing_conversations()
                if len(passing_conversations) > 0:
                    if self.verbose:
                        print("[LLMcoder] Code is correct. Stopping early...")
                    return self._get_best_completion(passing_conversations)

        # No conversation passes, so just return the best completion
        return self._get_best_completion(self.conversations)

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
        # Get the completions from OpenAI's API
        candidates = self.client.chat.completions.create(
            messages=conversation.messages,
            model=model,
            temperature=temperature,
            n=n)  # type: ignore

        # Count the number of tokens generated
        self.n_tokens_generated += sum([len(self.encoder.encode(message.message.content)) for message in candidates.choices])

        # Filter out completions that are repetitions of previous mistakes
        valid_choices = list(set([completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]))

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
            valid_choices = list(set([completion for completion in candidates.choices if not self._is_bad_completion(completion.message.content)]))

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
                print(f"[LLMcoder] Analyzing {len(valid_choices)} completions...")

            with ThreadPoolExecutor(max_workers=self.n_procs) as executor:
                # Create a mapping of future to completion choice
                choice_to_future = {
                    i: executor.submit(self._run_analyzers, conversation.messages[1]["content"], choice.message.content)
                    for i, choice in enumerate(valid_choices)}

                # Retrieve results in the order of valid_choices
                for i in range(len(valid_choices)):
                    future = choice_to_future[i]
                    try:
                        # Update the analyzer results history with the results of total completion
                        analysis = future.result()
                        analysis_score = sum([results["score"] for results in analysis.values()])

                        self.conversations.push(conversation
                                                .copy()
                                                .set_score(analysis_score)
                                                .add_analysis(analysis)
                                                .add_message({'role': 'assistant', 'content': valid_choices[i].message.content})
                                                .update_passing()
                                                .add_to_path(choice=i))
                    except Exception as e:
                        if self.verbose:
                            print(f"[LLMcoder] An exception occurred during analysis of choice: {e}")
                        raise e

        # If we only have one completion, still run the analyzers on it
        else:
            if self.verbose:
                print("[LLMcoder] Analyzing completion...")
            analysis = self._run_analyzers(conversation.messages[1]["content"], valid_choices[0].message.content)
            analysis_score = sum([results["score"] for results in analysis.values()])

            # There is only one valid choice
            # Update the analyzer results history with the results of the completion
            self.conversations.push(conversation
                                    .copy()
                                    .set_score(analysis_score)
                                    .add_analysis(analysis)
                                    .add_message({'role': 'assistant', 'content': valid_choices[0].message.content})
                                    .update_passing()
                                    .add_to_path(choice=0))

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
                    "content": self.system_prompt}]))

    def _run_analyzers(self, code: str, completion: str) -> dict[str, dict]:
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

    def _step(self, code: str, temperature: float = 0.7, meta_temperature: float = 0.0, n: int = 1) -> None:
        """
        Complete the provided code with the OpenAI model and feedback, if available
        Make choice on highest scored snippet through PriorityQueue.pop().

        Parameters
        ----------
        code : str
            The code to complete
        temperature : float, optional
            The temperature to use for the completion, by default 0.7
        meta_temperature : float, optional
            The temperature to use for choosing the most promising conversation, by default 0.1
        n : int, optional
            The number of choices to generate, by default 1
        """
        # Choose highest-scored conversation from the priority queue
        most_promising_conversation = self.conversations.pop(temperature=meta_temperature)

        if self.verbose:
            print(f'[LLMcoder] Choosing conversation {"-".join(str(node) for node in most_promising_conversation.path)} with score {round(most_promising_conversation.score, 2)}')

        # If there is feedback available from previous analyses, add it to the prompt
        if len(most_promising_conversation.analyses) > 0:
            feedback_prompt = self._feedback_prompt_template(
                [str(result['message']) for result in most_promising_conversation.analyses[-1].values()
                 if (not result['pass'] or result['type'] == "info")])

        # If there is not feedback available, the prompt will just be the user's code
        else:
            feedback_prompt = ""

        # Add the prompt to the messages
        most_promising_conversation.add_message({'role': 'user', 'content': feedback_prompt + code})

        # Get new completions and add them to the priority queue
        self._get_completions_for(most_promising_conversation, self.model_feedback, temperature, n)

        if self.verbose:
            probabilities = self.conversations.get_probabilities(temperature=meta_temperature)
            print(f'[LLMcoder] Have {len(self.conversations)} conversations:')
            print(f'[LLMcoder] {"Passing":<10}{"Score":<10}{"Prob":<10}Path')
            for c, prob in zip(self.conversations.queue, probabilities):
                print(f'[LLMcoder] {str(c.passing):<10}{round(c.score, 2):<10}{round(prob, 4):<10}{c.path}')

        self.iterations += 1
