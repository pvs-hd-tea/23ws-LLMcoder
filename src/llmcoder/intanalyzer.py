# Interface for definition of suggestions' analysis. 
# Inherited by 
# 1. SyntaxAnalyzer
# 2. UnitTestAnalyzer
# 3. APIDocumentationAnalyzer

"""
Python further deviates from other languages in one other aspect. 
It doesn’t require the class that’s implementing the interface to define all of the interface’s abstract methods.
"""

import os
import abc


# Interface to follow the Duck Typing principle supported by Python
# TODO: be defined a wrapper for the BaseDecorator
class FormalAnalyzerInterface(metaclass = abc.ABCMeta):
    """
    The hasattr() function checks if an attribute (including a method) exists, 
    and the callable() function checks if the attribute is callable (i.e., a method).
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'analyze') and 
                callable(subclass.analyze))
            
    # Required method for the analyzer to be implemented in concrete component
    # @abstractmethod
    def analyze(self, code_suggestion: str) -> str:
        pass
    

