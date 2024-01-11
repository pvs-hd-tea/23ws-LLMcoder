import os
import re
import tempfile

import jedi

from llmcoder.analyze.Analyzer import Analyzer


class JediAnalyzer(Analyzer):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        super().__init__(verbose)

    def analyze(self, input: str, completion: str, context: dict[str, dict[str, float | int | str]] | None = None) -> dict[str, object]:
        results = []

        code = input + completion

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code)

        # From the context (i.e. the results from the previous analyzers like the mypy_analyzer_v1), we can get problematic functions and classes
        # and use them as the query for the get_signature_and_doc function.
        query = []
        if context:
            if self.verbose:
                print(f"[JediAnalyzer] Using context from previous analyzers: {list(context.keys())}")
            if 'mypy_analyzer_v1' in context and isinstance(context['mypy_analyzer_v1']['message'], str):
                for line in context['mypy_analyzer_v1']['message'].split("\n"):
                    if line.startswith(f"{temp_file_name}:"):
                        # Extract the problematic function or class name from the mypy_analyzer_v1 result
                        # Mypy will wrap the name in quotation marks like "foo" if it is a function, and in quotation marks and parentheses like "Foo" if it is a class.
                        # Find the quotation marks and extract the name.
                        # E.g. from `your completion:6: error: Argument 1 to "Client" has incompatible type "str | None"; expected "str"  [arg-type] Found 1 error in 1 file (checked 1 source file)`extract the name "Client".

                        matches_has = re.findall(r'\"(.+?)\" has', line)
                        matches_for = re.findall(r'for \"(.+?)\"', line)
                        matches_gets = re.findall(r'\"(.+?)\" gets', line)
                        # TODO: There may be more

                        matches = matches_has + matches_for + matches_gets

                        for match in matches:
                            if self.verbose:
                                print(f"[JediAnalyzer] Found problematic function or class: {match}")

                            # Sometimes, the queries are "type[ElasticsearchStore]" instead of "ElasticsearchStore".
                            # Extract the class name from the query.
                            if match.startswith("type["):
                                match = match[5:-1]

                            query.append(match)

        # Remove duplicates
        query = list(set([q for q in query if q.strip() != ""]))
        print(query)

        # If there is no query, there is nothing to do
        if len(query) == 0:
            if self.verbose:
                print("[JediAnalyzer] No problematic functions or classes found in the context.")
            os.remove(temp_file_name)
            return {
                "pass": True,
                "type": "info",
                "score": 0,
                "message": "All functions and classes in your completion are called correctly (their signatures match with the documentation)."
            }

        # If there is a query, get the signatures and documentations of the functions and classes that match the query
        else:
            # script_input = input+completion
            script = jedi.Script(code=temp_file_name)
            names = script.get_names(all_scopes=True, definitions=True)

            for name in names:
                if name.full_name is None:
                    continue

                if "def" not in name.description and "class" not in name.description:
                    continue

                result = {
                    "name": name.full_name.split(".")[-1],
                    "signature": name.get_type_hint(),
                    "doc": name.docstring()
                }
                results.append(result)
            result_str = "To fix these errors, use these ground truth signatures as a reference for your next completions:\n"
            result_str += "\n".join([f"{result['name']}: {result['signature'] if result['signature'] else result['doc']}" for result in results])

        os.remove(temp_file_name)

        return {
            "pass": True,
            "type": "info",
            "score": -len(result),
            "message": result_str
        }
