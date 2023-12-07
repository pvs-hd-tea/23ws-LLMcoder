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
        if isinstance(query, str):
            query = [query]
        elif isinstance(query, list) and not query:
            print("Empty query specified.")
            return

        with open(path) as fh:
            root = ast.parse(fh.read(), path)

        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    module_name = alias.asname if alias.asname else alias.name
                    if query:
                        for q in query:
                            if q.startswith(module_name + ".") or q == module_name:
                                yield Import([module], q.split('.')[-1], module_name)
                    else:
                        yield Import([module], None, module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                for alias in node.names:
                    module = node.module.split('.')
                    name = alias.name
                    asname = alias.asname if alias.asname else name
                    if query:
                        if name in query or asname in query:
                            yield Import(module, name, asname)
                    else:
                        yield Import(module, name, asname)

        # TODO: Handle builtins

    def find_function_calls(self, code: str, query: str | list[str] | None):
        root = ast.parse(code)
        function_calls = []

        if isinstance(query, str):
            query = [query]

        for node in ast.walk(root):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Handle nested attributes
                    attribute_chain = []
                    current = node.func
                    while isinstance(current, ast.Attribute):
                        attribute_chain.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        attribute_chain.append(current.id)
                    attribute_chain.reverse()
                    module_alias = attribute_chain[0]
                    func_name = '.'.join(attribute_chain[1:])
                elif isinstance(node.func, ast.Name):
                    # Direct function call
                    module_alias = None
                    func_name = node.func.id
                else:
                    continue

                if not query or func_name in query or func_name.split(".")[-1] in query or '.'.join(attribute_chain) in query:
                    function_calls.append((module_alias, func_name))

        return function_calls

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
        import_aliases = {}
        direct_imports = {}  # Store direct imports

        with open(path) as file:
            code = file.read()

        # Get all imports
        for imp in self.get_imports(path):
            full_module = '.'.join(imp.module) if imp.module else imp.name[0]
            if imp.module and imp.name:  # Correctly identify direct imports
                direct_imports[imp.name] = full_module
            else:
                alias = imp.alias if imp.alias else imp.name[-1] if imp.name else full_module
                import_aliases[alias] = full_module

        print(f"{import_aliases=}")
        print(f"{direct_imports=}")

        # Find all function calls that match the query
        function_calls = self.find_function_calls(code, query)

        function_calls = list(set(function_calls))

        print(f"{function_calls=}")

        # Match the function calls to the imports
        matched_function_calls = []
        for module_alias, func_name in function_calls:
            if module_alias and module_alias in import_aliases:
                matched_function_calls.append((module_alias, func_name))
            elif func_name in direct_imports:
                matched_function_calls.append((direct_imports[func_name], func_name))
            else:
                print(f"No import found for {func_name}")

        for module_alias, func_name in matched_function_calls:
            print(f"{module_alias=} {func_name=}")
            try:
                if module_alias and module_alias in import_aliases:
                    module_path = import_aliases[module_alias]
                    parts = func_name.split('.')
                    module = __import__(module_path, fromlist=[parts[0]])
                    attr = module
                    for part in parts:
                        attr = getattr(attr, part, None)
                elif func_name in direct_imports:  # Handle direct imports
                    module_name = direct_imports[func_name]
                    module = __import__(module_name, fromlist=[func_name])
                    attr = getattr(module, func_name, None)
                else:
                    attr = None

                if attr and callable(attr):
                    try:
                        sig = inspect.signature(attr)
                        doc = inspect.getdoc(attr)
                        signature_and_doc.append({
                            "name": func_name,
                            "signature": str(sig),
                            "doc": doc
                        })
                    except ValueError:
                        signature_and_doc.append({
                            "name": func_name,
                            "signature": None,
                            "doc": inspect.getdoc(attr)
                        })
                else:
                    print(f"No callable attribute {func_name} found")

            except (ImportError, AttributeError) as e:
                print(f"Error importing {func_name}: {e}")

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
