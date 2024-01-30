"""
Implementation of Tree of Completions using Tree in Conversation,
using a Priority Queue implemented with a Binary Heaps to reduce time-complexity.
"""

import heapq


class Conversation:
    def __init__(self, score: int, messages: list[dict[str, str]], analyses: list[dict[str, dict[str, float | int | str | bool]]] | None = None):
        self.messages = messages
        self.score = score
        self.analyses = analyses or []

    def invert(self) -> "Conversation":
        return Conversation(
            - self.score,
            self.messages,
            self.analyses
        )

    # Help function for completion
    def add_message(self, message: dict[str, str]) -> bool:
        self.messages.append(message)
        return True

    # Help function for _get_completion_for
    def add_analysis(self, analysis_results: dict[str, dict[str, float | int | str | bool]]) -> bool:
        self.analyses.append(analysis_results)
        return True

    def get_last_message(self) -> str:
        return self.messages[-1]["content"]

    # Enabling comparison for conversations
    def __gt__(self, conversation2: "Conversation") -> bool:
        return self.score > conversation2.score

    def __ge_(self, conversation2: "Conversation") -> bool:
        return self.score >= conversation2.score

    def __lt__(self, conversation2: "Conversation") -> bool:
        return self.score < conversation2.score

    def __le__(self, conversation2: "Conversation") -> bool:
        return self.score >= conversation2.score


class PriorityQueue:
    def __init__(self, conversations: Conversation | list[Conversation] | None = None):
        self.queue: list[Conversation] = []

        if conversations is not None:
            if not isinstance(conversations, list):
                conversations = [conversations]

            for conversation in conversations:
                # Invert the score of the conversation (subject to maximization) for the min-heap
                self.push(conversation.invert())

    def push(self, conversation: Conversation) -> None:
        """
        Push a conversation to the priority queue

        Parameters
        ----------
        conversation : Conversation
            The conversation to be pushed to the priority queue
        """
        heapq.heappush(self.queue, conversation.invert())

    def pop(self, keep: bool = False) -> Conversation:
        """
        Pop a conversation from the priority queue

        Parameters
        ----------
        keep : bool
            Whether to keep the conversation in the queue

        Returns
        -------
        Conversation
            The conversation with the highest score
        """
        if keep:
            # Return the best conversation without removing it from the queue
            return self.queue[0].invert()

        # Pop and return the conversation with the highest score
        conversation = heapq.heappop(self.queue)
        return conversation

    def clear(self) -> bool:
        self.queue = []

    def __len__(self) -> int:
        return len(self.queue)
