# Interface for definition of suggestions' analysis.
# Inherited by
# 1. SyntaxAnalyzer
# 2. UnitTestAnalyzer
# 3. APIDocumentationAnalyzer

"""
Python further deviates from other languages in one other aspect.
It doesn’t require the class that’s implementing the interface to define all of the interface’s abstract methods.
"""

from abc import ABCMeta, abstractmethod


# Interface to follow the Duck Typing principle supported by Python
# TODO: be defined a wrapper for the BaseDecorator
class FormalAnalyzerInterface(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.code_suggestion = ''

    # Setter method for code_suggestion
    @classmethod
    def set_code_suggestion(self, code_suggestion: str) -> None:
        self.code_suggestion = code_suggestion

    # Getter method for code_suggestion
    @classmethod
    def get_code_suggestion(self) -> str:
        return self.code_suggestion

    # Required method for the analyzer to be implemented in concrete component
    # Can return an error from unittest,syntaxtest: str, return docs: list[dict[str, str]] from APIDocs, return None for nothing
    @abstractmethod
    def analyze(self, code_suggestion: str) -> str | list[dict[str, str]] | None:
        self.set_code_suggestion(code_suggestion)
        raise NotImplementedError("Analyze abstract method is not implemented!")
