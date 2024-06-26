from typing import Any


class Conversation:
    """
    A class to represent a conversation, which contains a list of messages, a score, and a list of analyses.

    Parameters
    ----------
    score : int
        The score of the conversation
    messages : list[dict[str, str]]
        The list of messages in the conversation
    analyses : list[dict[str, dict[str, float | int | str | bool]]] | None, optional
        The list of analyses in the conversation, by default None
    path : list[Any] | None, optional
        The path of the conversation in the conversation tree, by default None
    passing : bool, optional
        Whether the conversation has passed all critical analyzers, by default False
    """
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
        """
        Update the passing status of the conversation based on the critical analyzers

        Returns
        -------
        Conversation
            The conversation with the updated passing status
        """
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

    def __contains__(self, rhs_conversation: "Conversation") -> bool:
        # Check if the path of the rhs_conversation is a subpath of the path of the lhs_conversation
        if len(rhs_conversation.path) > len(self.path):
            return False

        for i in range(len(rhs_conversation.path)):
            if rhs_conversation.path[i] != self.path[i]:
                return False

        return True
