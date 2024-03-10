import re

import jedi
import Levenshtein
import openai
from langchain_community.embeddings import GPT4AllEmbeddings

from llmcoder.analyze.Analyzer import Analyzer
from llmcoder.utils import get_openai_key


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
        self.client = openai.OpenAI(api_key=get_openai_key())
        self.gpt4all = GPT4AllEmbeddings()

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
        code = input + completion
        script = jedi.Script(code)

        # From the context (i.e. the results from the previous analyzers like the mypy_analyzer_v1), we can get hallucinated functions or classes.
        hallucinations = []
        n_total_suggestions = 0
        if context:
            if self.verbose:
                print(f"[Hallucinations] Using context from previous analyzers: {list(context.keys())}")
            if 'mypy_analyzer_v1' in context and isinstance(context['mypy_analyzer_v1']['message'], str):
                for line in context['mypy_analyzer_v1']['message'].split("\n"):
                    if line.startswith("your completion:"):
                        error_line_number = int(line.split(":")[1])
                        error_line = code.split("\n")[error_line_number - 1]

                        matches_no_attr = re.findall(r'has no attribute \"(.+?)\"', line)
                        # TODO: There may be more

                        matches = matches_no_attr

                        for match in matches:
                            if self.verbose:
                                print(f"[Hallucinations] Found problematic function or class: {match}")

                            # Sometimes, the queries are "type[ElasticsearchStore]" instead of "ElasticsearchStore".
                            if match.startswith("type["):
                                match = match[5:-1]

                            module_matches = re.findall(r'([a-zA-Z0-9_]*)\.([a-zA-Z0-9_]*)', error_line)

                            if len(module_matches) > 0:
                                module_of_hallucinated_attribute = '.'.join(module_matches[0][:-1])

                                try:  # https://www.phind.com/search?cache=ey5i26k2mr5wuezcjx9tzkaf
                                    hallucinated_attribute = module_matches[0][-1]
                                    suggested_attributes = script.complete_search(module_of_hallucinated_attribute + '.')
                                    n_total_suggestions += len(suggested_attributes)

                                    suggested_attributes_distances = sorted([(Levenshtein.distance(hallucinated_attribute, suggested_attribute.name), suggested_attribute) for suggested_attribute in suggested_attributes if hasattr(suggested_attribute, "name") and isinstance(suggested_attribute.name, str)], key=lambda sim: sim[0])[:10]
                                    suggested_attributes = [suggested_attribute_distance[1] for suggested_attribute_distance in suggested_attributes_distances]

                                except AttributeError:
                                    continue
                            else:
                                module_of_hallucinated_attribute = None
                                suggested_attributes = []

                            if self.verbose:
                                print(f"[Hallucinations] Found hallucination: {match} in {module_of_hallucinated_attribute} with {len(suggested_attributes)} suggestions")
                            hallucinations.append({
                                'name': match,
                                'module': module_of_hallucinated_attribute,
                                'suggested_attributes': suggested_attributes
                            })

        # Remove duplicate hallucinations based on the `module.name` property
        hallucinations_dedupe = []
        hallucinations_full_names = []
        for hallucination in hallucinations:
            full_name = hallucination['module'] + '.' + hallucination['name']
            if full_name not in hallucinations_full_names:
                if self.verbose:
                    print(f"[Hallucinations] Full name: {full_name}, append to {hallucinations_dedupe}")
                hallucinations_full_names.append(full_name)
                hallucinations_dedupe.append(hallucination)

        # If there is no query, there is nothing to do
        if len(hallucinations) == 0:
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
                print(f"[Hallucinations] Found {len(hallucinations)} hallucinations: {[hallucination['module'] + '.' + hallucination['name'] for hallucination in hallucinations]}")

            results_str = "The following attributes do not exist:\n"
            for h in hallucinations_dedupe:
                results_str += f"- '{h['name']}' of '{h['module']}'\n"
            results_str += "\n"
            results_str += "Do not use these attributes."

            if n_total_suggestions > 0:
                results_str += "\n\n"
                for h in hallucinations_dedupe:
                    results_str += f"Instead of '{h['module']}.{h['name']}', use the most plausible of these attributes: {h['module']}.[" + ', '.join([s.name for s in h['suggested_attributes']]) + "]\n"
                    if self.verbose:
                        print(f"[Hallucinations] Suggested attributes for {h['module']}.{h['name']}: {[s.name for s in h['suggested_attributes']]}")

            n_hallucinations_without_suggestions = len([h for h in hallucinations_dedupe if len(h['suggested_attributes']) == 0])
            n_hallucinations_with_suggestions = len([h for h in hallucinations_dedupe if len(h['suggested_attributes']) > 0])

            return {
                "pass": False,
                "type": "info",
                "score": - n_hallucinations_without_suggestions + 0.5 * n_hallucinations_with_suggestions,
                "message": results_str
            }
