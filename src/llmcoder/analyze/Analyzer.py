from abc import ABC, abstractmethod


class Analyzer(ABC):
    # Required method for the analyzer to be implemented in concrete component

    def __init__(self) -> None:
        self.input = ""
        self.completion = ""

    @abstractmethod
    def analyze(self, input: str, completion: str) -> dict:
        raise NotImplementedError
