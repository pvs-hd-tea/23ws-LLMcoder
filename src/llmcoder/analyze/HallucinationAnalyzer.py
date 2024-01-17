import re

from llmcoder.analyze.Analyzer import Analyzer


class HallucinationAnalyzer(Analyzer):
    """
    Analyzer that checks mypy errors for hallucinations.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the SignatureAnalyzer.

        Parameters
        ----------
        verbose : bool
            Whether to print debug messages.
        """
        super().__init__(verbose)

    def analyze(self, input: str, completion: str, context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
        """
        Analyze the completion and return a message.

        Parameters
        ----------
        input : str
            The input code.
        completion : str
            The completion code.
        context : dict[str, dict[str, float | int | str]] | None
            The context from the previous analyzers.

        Returns
        -------
        dict
            A dictionary containing the result of the analysis.
        """
        # From the context (i.e. the results from the previous analyzers like the mypy_analyzer_v1), we can get hallucinated functions or classes.
        hallucinated_names = []
        if context:
            if self.verbose:
                print(f"[Hallucinations] Using context from previous analyzers: {list(context.keys())}")
            if 'mypy_analyzer_v1' in context and isinstance(context['mypy_analyzer_v1']['message'], str):
                for line in context['mypy_analyzer_v1']['message'].split("\n"):
                    if line.startswith("your completion:"):
                        # Extract the problematic function or class name from the mypy_analyzer_v1 result
                        # Mypy will wrap the name in quotation marks like "foo" if it is a function, and in quotation marks and parentheses like "Foo" if it is a class.
                        # Find the quotation marks and extract the name.
                        # E.g. from `your completion:6: error: Module has no attribute "TikTok"  [attr-defined]

                        matches_no_attr = re.findall(r'has no attribute \"(.+?)\"', line)
                        # TODO: There may be more

                        matches = matches_no_attr

                        for match in matches:
                            if self.verbose:
                                print(f"[Hallucinations] Found problematic function or class: {match}")

                            # Sometimes, the queries are "type[ElasticsearchStore]" instead of "ElasticsearchStore".
                            if match.startswith("type["):
                                match = match[5:-1]

                            hallucinated_names.append(match)

        # Remove duplicates
        hallucinated_names = list(set([name for name in hallucinated_names if name.strip() != ""]))

        # If there is no query, there is nothing to do
        if len(hallucinated_names) == 0:
            if self.verbose:
                print("[Hallucinations] No hallucinations found.")
            return {
                "pass": True,
                "type": "info",
                "score": 0,
                "message": ""  # No message
            }

        else:
            if self.verbose:
                print(f"[Hallucinations] Found {len(hallucinated_names)} hallucinations: {hallucinated_names}")

            results_str = "You used the following non-existent functions or classes:\n"
            for name in hallucinated_names:
                results_str += f"- {name}\n"
            results_str += "\n"
            results_str += "In your next completion, do not use these functions or classes."

            return {
                "pass": False,
                "type": "info",
                "score": - len(hallucinated_names),
                "message": results_str
            }
