from typing import Any


class Conversation:
    def __init__(
            self,
            score: int,
            messages: list[dict[str, str]],
            analyses: list[dict[str, dict[str, float | int | str | bool]]] | None = None,
            path: list[Any] | None = None,
            passing: bool = False):
        self.messages = messages
        self.score = score
        self.passing = passing
        self.analyses = analyses or []
        self.path = path or ['R']

    def copy(self) -> "Conversation":
        return Conversation(
            self.score,
            self.messages.copy(),
            self.analyses.copy(),
            self.path.copy(),
            self.passing
        )

    def invert_score(self) -> "Conversation":
        return Conversation(
            - self.score,
            self.messages.copy(),
            self.analyses.copy(),
            self.path.copy(),
            self.passing
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

    def update_passing(self) -> "Conversation":
        # Print how many critical analyzers have passed
        n_passed = sum(results['pass'] for results in self.analyses[-1].values() if (results['type'] == "critical" and type(results['pass']) is bool))
        n_total = len([results for results in self.analyses[-1].values() if results['type'] == "critical" and type(results['pass']) is bool])

        self.passing = (n_passed == n_total)

        return self

    def get_last_message(self) -> str:
        return self.messages[-1]["content"]

    # Enabling comparison for conversations
    def __gt__(self, rhs_conversation: "Conversation") -> bool:
        return self.score > rhs_conversation.score

    def __ge_(self, rhs_conversation: "Conversation") -> bool:
        return self.score >= rhs_conversation.score

    def __lt__(self, rhs_conversation: "Conversation") -> bool:
        return self.score < rhs_conversation.score

    def __le__(self, rhs_conversation: "Conversation") -> bool:
        return self.score <= rhs_conversation.score
