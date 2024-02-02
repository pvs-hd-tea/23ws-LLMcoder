from typing import Iterator

from llmcoder.conversation import Conversation
import numpy as np


class PriorityQueue:
    def __init__(self, conversations: Conversation | list[Conversation] | None = None):
        self.queue: list[Conversation] = []

        if conversations is not None:
            self.push(conversations)

    def push(self, conversation: Conversation | list[Conversation]) -> None:
        """
        Push a conversation to the priority queue and sort it

        Parameters
        ----------
        conversation : Conversation | list[Conversation]
            The conversation to be pushed to the priority queue
        """
        if isinstance(conversation, list):
            self.queue.extend(conversation)
        else:
            self.queue.append(conversation)

        # Sort the queue after each insertion
        self.queue.sort(reverse=True)

    def pop(self, temperature: float = 0, keep: bool = False) -> Conversation:
        """
        Pop a conversation from the priority queue

        Parameters
        ----------
        keep : bool
            Whether to keep the conversation in the queue
        temperature : float
            Temperature parameter for softmax, default is 0

        Returns
        -------
        Conversation
            The conversation with the highest score
        """
        # If temperature is 0, return the conversation with the highest score
        if temperature == 0:
            index = 0
        # If the temerature is infinite, sample uniformly
        elif temperature == np.inf:
            index = np.random.choice(len(self.queue))
        # Otherwise, sample from the queue using softmax
        else:
            index = self.sample(temperature=temperature)

        # If keep, return the conversation without removing it from the queue
        if keep:
            return self.queue[index]

        # Otherwise, return the conversation and remove it from the queue
        return self.queue.pop(index)

    def sample(self, temperature: float = 1.0) -> int:
        """
        Sample a conversation from the priority queue using softmax

        Parameters
        ----------
        temperature : float
            Temperature parameter for softmax, default is 1.0

        Returns
        -------
        int
            The index of the sampled conversation
        """
        # Collect all scores and compute the softmax probabilities
        scores = [c.score for c in self.queue]
        probs = softmax(scores, temperature)

        # Sample from the queue using the computed probabilities
        return np.random.choice(len(self.queue), p=probs)

    def clear(self) -> "PriorityQueue":
        self.queue = []
        return self

    def __len__(self) -> int:
        return len(self.queue)

    def __iter__(self) -> Iterator[Conversation]:
        return self.queue.__iter__()


def softmax(x: list | np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute the softmax function for an input array x with a given temperature.

    Parameters
    ----------
    x : numpy.ndarray
        Input array
    temperature : float, optional
        Temperature parameter for softmax, default is 1.0

    Returns
    -------
    numpy.ndarray
        Softmax output array
    """
    # Normalize the input array using the temperature parameter and the maximum value
    e_x = np.exp(x / temperature - np.max(x / temperature))
    return e_x / np.sum(e_x)
