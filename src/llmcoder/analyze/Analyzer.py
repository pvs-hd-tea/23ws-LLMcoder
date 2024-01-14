from abc import ABC, abstractmethod


class Analyzer(ABC):
    """
    Abstract class for analyzers

    Attributes
    ----------
    verbose : bool
        Whether to print out debug information
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def setup(self, input: str) -> None:
        """
        Setup the analyzer with the input string for caching or other purposes

        Parameters
        ----------
        input : str
            Input string to be analyzed
        """
        pass

    @abstractmethod
    def analyze(self, input: str, completion: str, context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
        raise NotImplementedError
