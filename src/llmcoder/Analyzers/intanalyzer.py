from abc import ABCMeta, abstractmethod


<<<<<<< HEAD:src/llmcoder/intanalyzer.py
class Analyzer(ABCMeta):

=======
# Interface to follow the Duck Typing principle supported by Python
# TODO: be defined a wrapper for the BaseDecorator
class FormalAnalyzerInterface(metaclass = ABCMeta):
    
>>>>>>> fetch-pak-docs:src/llmcoder/Analyzers/intanalyzer.py
    # Required method for the analyzer to be implemented in concrete component
    @abstractmethod
    def analyze(self, code_suggestion: str) -> dict:
        pass
