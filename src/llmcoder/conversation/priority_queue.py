import heapq
from typing import Iterator

from llmcoder.conversation import Conversation


class PriorityQueue:
    def __init__(self, conversations: Conversation | list[Conversation] | None = None):
        self.queue: list[Conversation] = []

        if conversations is not None:
            if not isinstance(conversations, list):
                conversations = [conversations]

            for conversation in conversations:
                # Invert the score of the conversation (subject to maximization) for the min-heap
                self.push(conversation)

    def push(self, conversation: Conversation) -> None:
        """
        Push a conversation to the priority queue

        Parameters
        ----------
        conversation : Conversation
            The conversation to be pushed to the priority queue
        """
        heapq.heappush(self.queue, conversation.invert_score())
        self.queue.sort()  # HACK: Reduntant but nice to have in the logs and not too slow for the number of conversations we consider

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
            return self.queue[0].invert_score()

        # Pop and return the conversation with the highest score
        conversation = heapq.heappop(self.queue)
        return conversation.invert_score()

    def clear(self) -> "PriorityQueue":
        self.queue = []
        return self

    def __len__(self) -> int:
        return len(self.queue)

    def __iter__(self) -> Iterator[Conversation]:
        return self.queue.__iter__()
