import ast
import inspect
import os
import re
# import builtins
import tempfile
from collections import namedtuple
from typing import Generator

from llmcoder.analyze.Analyzer import Analyzer

Import = namedtuple("Import", ["module", "name", "alias"])


class SignatureAnalyzer(Analyzer):

    def get_imports(self, path: str, query: str | list[str] | None = None) -> Generator:
        """
        Get all imports from a Python file that match the query, if specified.

        Parameters
        ----------
        path : str
            Path to the Python file. Can be temporary.
        query : str | list[str] | None
            The query string to search for. E.g. a function name or a class name.

        Returns
        -------
        Generator
            A generator that yields Import objects that match the query, if specified.
        """
        if isinstance(query, str):
            query = [query]

        elif isinstance(query, list):
            if len(query) == 0:
                print("Empty query specified.")
                return

        with open(path) as fh:
            root = ast.parse(fh.read(), path)

        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                module = []
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                module = node.module.split('.')
            else:
                continue

            for n in node.names:  # type: ignore
                if query:
                    if n.name in query:
                        print(f"Name {n.name} matches query {query}")
                        yield Import(module, n.name.split('.'), n.asname)
                else:
                    print(f"Name {n.name} matches wildcard query")
                    yield Import(module, n.name.split('.'), n.asname)

        # TODO: Handle builtins

    def get_signature_and_doc(self, path: str, query: str | list[str] | None) -> list[dict]:
        """
        Get the signature and documentation of a function or class.

        Parameters
        ----------
        path : str
            Path to the Python file. Can be temporary.
        query : str | list[str] | None
            The query string to search for. E.g. a function name or a class name.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the signature and documentation of every match to the query.
        """
        signature_and_doc = []

        for imp in self.get_imports(path, query):
            if imp.module:
                module = '.'.join(imp.module)
            else:
                module = imp.name[0]

            name = imp.alias if imp.alias else imp.name[0]

            try:
                obj = __import__(module, fromlist=[name])
                obj = getattr(obj, name)

                try:
                    sig = inspect.signature(obj)  # type: ignore
                    doc = inspect.getdoc(obj)
                except ValueError:
                    # Built-in function
                    sig = inspect.signature(obj.__call__)
                    doc = inspect.getdoc(obj.__call__)

                signature_and_doc.append({
                    "name": name,
                    "signature": str(sig),
                    "doc": doc
                })

            except (ImportError, AttributeError):
                print(f"Cannot get signature and documentation of {name}")
                continue

        return signature_and_doc

    def analyze(self, input: str, completion: str, context: dict[str, dict[str, bool | str]] | None = None) -> dict:
        code = input + completion

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code)

        # From the context (i.e. the results from the previous analyzers like the mypy_analyzer_v1), we can get problematic functions and classes
        # and use them as the query for the get_signature_and_doc function.
        query = []
        if context:
            print(f"Using context from previous analyzers: {list(context.keys())}")
            if 'mypy_analyzer_v1' in context and isinstance(context['mypy_analyzer_v1']['message'], str):
                for line in context['mypy_analyzer_v1']['message'].split("\n"):
                    if line.startswith("your completion:"):
                        # Extract the problematic function or class name from the mypy_analyzer_v1 result
                        # Mypy will wrap the name in quotation marks like "foo" if it is a function, and in quotation marks and parentheses like "Foo" if it is a class.
                        # Find the quotation marks and extract the name.
                        # E.g. from `your completion:6: error: Argument 1 to "Client" has incompatible type "str | None"; expected "str"  [arg-type] Found 1 error in 1 file (checked 1 source file)`extract the name "Client".

                        matches_has = re.findall(r'\"(.+?)\" has', line)
                        matches_for = re.findall(r'for \"(.+?)\"', line)

                        matches = matches_has + matches_for

                        for match in matches:
                            print(f"Found problematic function or class: {match}")

                            # Sometimes, the queries are "type[ElasticsearchStore]" instead of "ElasticsearchStore".
                            # Extract the class name from the query.
                            if match.startswith("type["):
                                match = match[5:-1]

                            query.append(match)

        query = list(set([q for q in query if q.strip() != ""]))

        if len(query) == 0:
            print("No problematic functions or classes found in the context.")
            os.remove(temp_file_name)
            return {
                "pass": "info",
                "message": "All functions and classes in your completion are called correctly (their signatures match with the documentation)."
            }
        else:
            result = self.get_signature_and_doc(temp_file_name, list(set(query)))

            print("Got signatures:")
            for r in result:
                print(r['signature'])

            if len(result) == 0:
                os.remove(temp_file_name)
                return {
                    "pass": "info",
                    "message": "Cannot find the relevant signatures of " + ", ".join(query)
                }

            result_str = "To fix these errors, use these ground truth signatures as a reference for your next completion:\n"
            result_str += "\n".join([f"{r['name']}: {r['signature']}" for r in result])

            os.remove(temp_file_name)
            return {
                "pass": "info",
                "message": result_str
            }
