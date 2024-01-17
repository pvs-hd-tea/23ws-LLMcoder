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
            case "gpt_review_analyzer_v1":
                raise DeprecationWarning("GPTReviewAnalyzer_v1 is deprecated")
            case "mypy_analyzer_v1":
                from llmcoder.analyze.MypyAnalyzer import MypyAnalyzer
                return MypyAnalyzer(*args, **kwargs)
            case "signature_analyzer_v1":
                from llmcoder.analyze.SignatureAnalyzer import SignatureAnalyzer
                return SignatureAnalyzer(*args, **kwargs)
            case "hallucination_analyzer_v1":
                from llmcoder.analyze.HallucinationAnalyzer import HallucinationAnalyzer
                return HallucinationAnalyzer(*args, **kwargs)
            case "gpt_score_analyzer_v1":
                from llmcoder.analyze.GPTScoreAnalyzer import GPTScoreAnalyzer
                return GPTScoreAnalyzer(*args, **kwargs)
            case _:
                raise ValueError(f"Invalid analyzer name: {analyzer}")
