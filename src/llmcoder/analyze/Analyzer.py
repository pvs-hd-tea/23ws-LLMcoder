from abc import ABC, abstractmethod


class Analyzer(ABC):
    # Required method for the analyzer to be implemented in concrete component

    def __init__(self) -> None:
        self.input = ""
        self.completion = ""

    @abstractmethod
    def analyze(self, input: str, completion: str, context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
        raise NotImplementedError
