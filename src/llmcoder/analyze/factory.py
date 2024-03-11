from typing import Any

from llmcoder.analyze.analyzer import Analyzer


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
                from llmcoder.analyze.mypy_analyzer import MypyAnalyzer
                return MypyAnalyzer(*args, **kwargs)
            case "signature_analyzer_v1":
                from llmcoder.analyze.signature_analyzer import SignatureAnalyzer
                return SignatureAnalyzer(*args, **kwargs)
            case "hallucination_analyzer_v1":
                from llmcoder.analyze.hallucination_analyzer import HallucinationAnalyzer
                return HallucinationAnalyzer(*args, **kwargs)
            case "gpt_score_analyzer_v1":
                from llmcoder.analyze.gpt_score_analyzer import GPTScoreAnalyzer
                return GPTScoreAnalyzer(*args, **kwargs)
            case "jedi_analyzer_v1":
                from llmcoder.analyze.jedi_analyzer import JediAnalyzer
                return JediAnalyzer(*args, **kwargs)
            case _:
                raise ValueError(f"Invalid analyzer name: {analyzer}")
