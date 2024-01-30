"""
Implementation of Tree of Completions using Tree in Conversation,
using a Priority Queue implemented with a Binary Heaps to reduce time-complexity.
"""

import heapq


class Conversation:
    def __init__(self, score: int, messages: list[dict[str, str]], analyzer_results_history: list[dict[str, dict[str, float | int | str | bool]]] = []):
        self.messages = messages
        self.score = score
        self.analyzer_results_history = analyzer_results_history

    # Show the score of the completion in positive numbers
    def _get_score(self) -> int:
        return -self.score

    def _get_messages(self) -> list[dict[str, str]]:
        return self.messages

    def _get_analyzer_results_history(self) -> list[dict[str, dict[str, float | int | str | bool]]]:
        return self.analyzer_results_history

    # Update the real score to store it as a "Min-heap"
    def _invert_score(self) -> None:
        inverted_score = -self.score
        self.score = inverted_score

    # Help function for completion
    def _add_message(self, message: dict[str, str]) -> bool:
        self.messages.append(message)
        return True

    # Help function for _get_completion_for
    def _add_analysis(self, analysis_results: dict[str, dict[str, float | int | str | bool]]) -> bool:
        self.analyzer_results_history.append(analysis_results)
        return True

    # Help function for _get_completion_for
    def _update_score(self, score: int) -> bool:
        self.score = -score
        return True

    def _get_last_message(self) -> str:
        return self.messages[-1]["content"]

    # Enabling comparison for conversations
    def __gt__(self, conversation2) -> bool:
        return self.score > conversation2.score

    def __ge_(self, conversation2) -> bool:
        return self.score >= conversation2.score

    def __lt__(self, conversation2) -> bool:
        return self.score < conversation2.score

    def __le__(self, conversation2) -> bool:
        return self.score >= conversation2.score


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def create_priority_queue(self, conversations: list[Conversation]) -> None:
        for conversation in conversations:
            # Multiply by -1 to trat as a "min-heap"
            conversation._invert_score()
            # We have to push the arguments separating them manually, since the score is needed for the priority
            self.push(conversation)

    def push(self, conversation: Conversation) -> None:
        heapq.heappush(self.queue, conversation)

    def pop(self) -> Conversation:
        # Pop and return the conversation with the highest score
        conversation = heapq.heappop(self.queue)
        return conversation

    def get_highest_scored_conversation(self) -> Conversation:
        # Keep but return the conversation with the highest score
        conversation = self.queue[0]
        return conversation

    def empty_queue(self) -> bool:
        while self.queue:
            self.pop()
        return True
    
    def __len__(self) -> int:
        return len(self.queue)