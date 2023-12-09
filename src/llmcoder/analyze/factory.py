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
            # case "syntax_analyzer_v1":
            #     from llmcoder.analyze.SyntaxAnalyzer import SyntaxAnalyzer
            #     return SyntaxAnalyzer(*args, **kwargs)
            # case "unit_test_analyzer_v1":
            #     from llmcoder.analyze.UnitTestAnalyzer import UnitTestAnalyzer
            #     return UnitTestAnalyzer(*args, **kwargs)
            case "gpt_review_analyzer_v1":
                raise DeprecationWarning("GPTReviewAnalyzer_v1 is deprecated")
                # from llmcoder.analyze.GPTReviewAnalyzer import GPTReviewAnalyzer_v1
                # return GPTReviewAnalyzer_v1(system_prompt="2023-12-02_GPTReviewAnalyzer_v4.txt")
            case "mypy_analyzer_v1":
                from llmcoder.analyze.MypyAnalyzer import MypyAnalyzer
                return MypyAnalyzer(*args, **kwargs)
            case "signature_analyzer_v1":
                from llmcoder.analyze.SignatureAnalyzer import SignatureAnalyzer
                return SignatureAnalyzer(*args, **kwargs)
            case "gpt_score_analyzer_v1":
                from llmcoder.analyze.GPTScoreAnalyzer import GPTScoreAnalyzer
                return GPTScoreAnalyzer(*args, **kwargs)
            case _:
                raise ValueError(f"Invalid analyzer name: {analyzer}")
