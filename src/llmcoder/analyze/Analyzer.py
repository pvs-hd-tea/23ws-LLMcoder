from abc import ABCMeta, abstractmethod


class Analyzer(ABCMeta):
    # Required method for the analyzer to be implemented in concrete component
    @abstractmethod
    def analyze(self, code: str) -> dict:
        pass
