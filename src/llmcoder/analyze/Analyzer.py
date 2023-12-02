from abc import ABC, abstractmethod


class Analyzer(ABC):
    # Required method for the analyzer to be implemented in concrete component
    @abstractmethod
    def analyze(self, input: str, completion: str) -> dict:
        pass
