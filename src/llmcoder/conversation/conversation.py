from typing import Any


class Conversation:
    def __init__(self, score: int, messages: list[dict[str, str]], analyses: list[dict[str, dict[str, float | int | str | bool]]] | None = None, path: list[Any] | None = None):
        self.messages = messages
        self.score = score
        self.analyses = analyses or []
        self.path = path or []

    def invert_score(self) -> "Conversation":
        return Conversation(
            - self.score,
            self.messages.copy(),
            self.analyses.copy()
        )

    # Help function for completion
    def add_message(self, message: dict[str, str]) -> "Conversation":
        self.messages.append(message)
        return self

    # Help function for _get_completion_for
    def add_analysis(self, analysis_results: dict[str, dict[str, float | int | str | bool]]) -> "Conversation":
        self.analyses.append(analysis_results)
        return self

    def set_score(self, score: int) -> "Conversation":
        self.score = score
        return self

    def add_to_path(self, choice: Any) -> "Conversation":
        self.path.append(choice)
        return self

    def get_last_message(self) -> str:
        return self.messages[-1]["content"]

    def copy(self) -> "Conversation":
        return Conversation(
            self.score,
            self.messages.copy(),
            self.analyses.copy()
        )

    # Enabling comparison for conversations
    def __gt__(self, rhs_conversation: "Conversation") -> bool:
        return self.score > rhs_conversation.score

    def __ge_(self, rhs_conversation: "Conversation") -> bool:
        return self.score >= rhs_conversation.score

    def __lt__(self, rhs_conversation: "Conversation") -> bool:
        return self.score < rhs_conversation.score

    def __le__(self, rhs_conversation: "Conversation") -> bool:
        return self.score <= rhs_conversation.score
