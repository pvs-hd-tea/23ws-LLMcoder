from typing import Any

from llmcoder.analyze.Analyzer import Analyzer


class AnalyzerFactory:
    """
    Factory class for Analyzer
    """
    @staticmethod
    def create_analyzer(analyzer: str, *args: Any, **kwargs: Any) -> Analyzer:
        """
        Create an analyzer from a string

        Parameters
        ----------
        analyzer : str
            The name of the analyzer to create

        Returns
        -------
        Analyzer
            The created analyzer
        """
        match analyzer:
            case "api_documentation_analyzer_v1":
                from llmcoder.analyze.APIDocumentationAnalyzer import APIDocumentationAnalyzer
                return APIDocumentationAnalyzer(*args, **kwargs)
            case "syntax_analyzer_v1":
                from llmcoder.analyze.SyntaxAnalyzer import SyntaxAnalyzer
                return SyntaxAnalyzer(*args, **kwargs)
            case "unit_test_analyzer_v1":
                from llmcoder.analyze.UnitTestAnalyzer import UnitTestAnalyzer
                return UnitTestAnalyzer(*args, **kwargs)
            case _:
                raise ValueError("Invalid analyzer name")
