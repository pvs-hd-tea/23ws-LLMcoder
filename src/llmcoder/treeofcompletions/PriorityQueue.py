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
    
    def _get_messages(self) -> list[str]:
        return self.messages
    
    def _get_analyzer_results_history(self):
        return self.analyzer_results_history
    
    # Update the real score to store it as a "Min-heap"
    def _invert_score(self):
        inverted_score = -self.score
        self.score = inverted_score

    # Help function for completion
    def _add_message(self, message) -> bool:
        self.messages.append(message)
        return True
    
    # Help function for _get_completion_for
    def _add_analysis(self, analysis_results: str) -> bool:
        analysis_results_list = []
        analysis_results_list.append(analysis_results)
        self.analyzer_results_history.append(analysis_results_list)
        return True
    
    # Help function for _get_completion_for
    def _update_score(self, score: int) -> bool:
        self.score = -score
        return True
    
    def _get_last_message(self) -> str:
        return self.messages[-1]["content"]

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def create_priority_queue(self, conversations: list[Conversation]):
        for conversation in conversations:
            # Multiply by -1 to trat as a "min-heap"
            conversation._invert_score()
            # We have to push the arguments separating them manually, since the score is needed for the priority
            self.push(conversation)
    
    def push(self, conversation: Conversation):
        heapq.heappush(self.queue, conversation)

    def pop(self) -> Conversation:
        # Pop and return the conversation with the highest score
        conversation = heapq.heappop(self.queue)
        return conversation
    
    def get_highest_scored_conversation(self) -> Conversation:
        # Keep but return the conversation with the highest score
        conversation = self.queue[0]
        return conversation
    
    def get_queue(self) -> heapq:
        return self.queue
    
    def empty_queue(self):
        while self.queue:
            conversation = self.pop(self.queue)
            # print(f'Score: {conversation._get_score()}, Text: {conversation._get_messages()}')

    
