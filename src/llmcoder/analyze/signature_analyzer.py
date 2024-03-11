import ast
import importlib
import inspect
import os
import re
import tempfile
from collections import namedtuple
from typing import Generator

from llmcoder.analyze.analyzer import Analyzer

Import = namedtuple("Import", ["module", "name", "alias"])


class SignatureAnalyzer(Analyzer):
    """
    Analyzer that fetches the signatures and documentations of functions and classes in the code.

    Parameters
    ----------
    verbose : bool
        Whether to print debug messages.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose)

    def get_imports(self, path: str, query: str | list[str] | None = None) -> Generator:
        """
        Find and yield all imports in the code.

        Parameters
        ----------
        path : str
            Path to the Python file. Can be temporary.
        query : str | list[str] | None
            The query string to search for. E.g. a function name or a class name.

        Yields
        ------
        Generator
            A generator that yields Import objects.
        """
        # Convert the query to a list if it is a string for convenience
        if isinstance(query, str):
            query = [query]

        # If the query is an empty list, return
        elif isinstance(query, list) and not query:
            if self.verbose:
                print("[Signatures] Empty query specified.")
            return

        # Parse the code with the ast module
        # FIXME: This will fail if the code is invalid
        with open(path) as fh:
            root = ast.parse(fh.read(), path)

        # Walk through the AST and find all imports
        for node in ast.walk(root):
            # Found an import statement
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name

                    # Set the module_name to the alias if it exists, otherwise to the module name
                    module_name = alias.asname if alias.asname else alias.name
                    if query:
                        for q in query:
                            # Check if the query is a module name or a function name
                            if q.startswith(module_name + ".") or q == module_name:
                                yield Import([module], q.split('.')[-1], module_name)

                    # In case ther is no query, yield all imports
                    else:
                        yield Import([module], None, module_name)

            # Found an import from statement
            elif isinstance(node, ast.ImportFrom):
                # If there is no module, this is a relative import
                # FIXME: Find a way to handle relative imports
                if node.module is None:
                    continue

                for alias in node.names:
                    module = node.module.split('.')  # type: ignore
                    name = alias.name

                    # Set the module_name to the alias if it exists, otherwise to the module name
                    asname = alias.asname if alias.asname else name
                    if query:
                        # Match the query to the module name or asname
                        if name in query or asname in query:
                            yield Import(module, name, asname)

                    # In case ther is no query, yield all imports
                    else:
                        yield Import(module, name, asname)

        # TODO: Handle builtins

    def find_function_calls(self, code: str, query: str | list[str] | None) -> list[tuple[str | None, str, str | None]]:
        """
        Find all function calls in the code that match the query.

        Parameters
        ----------
        code : str
            The code to analyze.
        query : str | list[str] | None
            The query string to search for. E.g. a function name or a class name.

        Returns
        -------
        list[tuple[str | None, str]]
            A list of tuples containing the module alias and the function name of every match to the query.
        """

        # Parse the code with the ast module
        root = ast.parse(code)
        function_calls: list[tuple[str | None, str, str | None]] = []
        aliases: dict[str, str | None] = {}

        # Convert the query to a list if it is a string for convenience
        if isinstance(query, str):
            query = [query]

        # Process import statements and assignments to track aliases
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for name in node.names:
                    aliases[name.asname or name.name] = name.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    aliases[name.asname or name.name] = module + '.' + name.name
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = self._resolve_alias(node.value, aliases)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Store the variable and the right-hand side expression
                        variable_name = target.id
                        rhs_expression = self._resolve_alias(node.value, aliases)
                        if rhs_expression:
                            aliases[variable_name] = rhs_expression

        # Walk through the AST and find all function calls
        for node in ast.walk(root):
            if isinstance(node, ast.Call):
                fully_qualified_func_name = self._resolve_function_call(node, aliases)

                # Extract the relevant part from the fully qualified name
                if fully_qualified_func_name:
                    parts = fully_qualified_func_name.split('.')

                    name_without_attribute = parts[-1]

                    # Check if the relevant name matches the query
                    if not query or name_without_attribute in query:
                        module_alias = parts[0] if len(parts) > 2 else None
                        function_calls.append((module_alias, name_without_attribute, None))

                    if len(parts) > 1:
                        name_with_attribute = '.'.join(parts[-2:])

                        # Check if the relevant name matches the query
                        if not query or name_with_attribute in query:
                            module_alias = parts[0] if len(parts) > 2 else None
                            function_calls.append((module_alias, parts[-2], parts[-1]))

        return function_calls

    def _resolve_function_call(self, node: ast.Call, aliases: dict[str, str | None]) -> str | None:
        if isinstance(node.func, ast.Attribute):
            resolved_chain = self._resolve_attribute_chain(node.func, aliases)
            if None in resolved_chain:
                return None
            return '.'.join(resolved_chain)  # type: ignore
        elif isinstance(node.func, ast.Name):
            return aliases.get(node.func.id, node.func.id)
        else:
            return None

    def _resolve_attribute_chain(self, node: ast.Attribute, aliases: dict[str, str | None]) -> list[str | None]:
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value  # type: ignore
        if isinstance(node, ast.Name):
            resolved_name = aliases.get(node.id, node.id)
            if resolved_name is None:
                return [None]
            chain.append(resolved_name)
        return list(reversed(chain))

    def _resolve_alias(self, node: ast.AST, aliases: dict[str, str | None]) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            resolved_chain = self._resolve_attribute_chain(node, aliases)
            return '.'.join(resolved_chain) if None not in resolved_chain else None  # type: ignore
        elif isinstance(node, ast.Call):
            # Handle the case where the alias is a result of a function call
            return self._resolve_function_call(node, aliases)
        return None

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

        # Read the code
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

        # Find all function calls that match the query
        function_calls = self.find_function_calls(code, query)

        print(f"[Signatures] {function_calls=}")

        function_calls = list(set(function_calls))

        # Match the function calls to the imports
        matched_function_calls = []
        for module_alias, func_name, attr_name in function_calls:
            if module_alias and module_alias in import_aliases:
                matched_function_calls.append((module_alias, func_name, attr_name))
            elif func_name in direct_imports:
                matched_function_calls.append((direct_imports[func_name], func_name, attr_name))
            else:
                if self.verbose:
                    print(f"[Signatures] No import found for {func_name}")

        for entry in matched_function_calls:
            # Parse the entry which could be a tuple of (module, class/function) or (module, class, attribute)
            module_alias, class_or_func, attribute = entry

            try:
                # Import the module
                module = importlib.import_module(import_aliases.get(module_alias, module_alias))
                # Resolve the class or function
                cls_or_func = getattr(module, class_or_func, None)
                if attribute:
                    # Resolve the attribute (method) if present
                    cls_or_func = getattr(cls_or_func, attribute, None)

                if cls_or_func and callable(cls_or_func):
                    try:
                        sig = inspect.signature(cls_or_func)
                        doc = inspect.getdoc(cls_or_func)
                        signature_and_doc.append({
                            "name": f"{class_or_func}.{attribute}" if attribute else class_or_func,
                            "signature": str(sig),
                            "doc": doc
                        })
                    except ValueError:
                        signature_and_doc.append({
                            "name": f"{class_or_func}.{attribute}" if attribute else class_or_func,
                            "signature": None,
                            "doc": inspect.getdoc(cls_or_func)
                        })
                else:
                    if self.verbose:
                        print(f"[Signatures] No callable attribute {class_or_func}.{attribute} found")

            except (ImportError, AttributeError) as e:
                if self.verbose:
                    print(f"[Signatures] Error importing {class_or_func}.{attribute}: {e}")

        return signature_and_doc

    def analyze(self, input: str, completion: str, context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
        """
        Analyze the completion and return a message.

        Parameters
        ----------
        input : str
            The input code.
        completion : str
            The completion code.
        context : dict[str, dict[str, float | int | str]] | None, optional
            The context of previous analyzers of the completion.

        Returns
        -------
        dict
            A dictionary containing the result of the analysis.
        """

        code = input + completion

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code)

        # From the context (i.e. the results from the previous analyzers like the mypy_analyzer_v1), we can get problematic functions and classes
        # and use them as the query for the get_signature_and_doc function.
        query = []
        if context:
            if self.verbose:
                print(f"[Signatures] Using context from previous analyzers: {list(context.keys())}")
            if 'mypy_analyzer_v1' in context and isinstance(context['mypy_analyzer_v1']['message'], str):
                for line in context['mypy_analyzer_v1']['message'].split("\n"):
                    if line.startswith("your completion:"):
                        # Extract the problematic function or class name from the mypy_analyzer_v1 result
                        # Mypy will wrap the name in quotation marks like "foo" if it is a function, and in quotation marks and parentheses like "Foo" if it is a class.
                        # Find the quotation marks and extract the name.
                        # E.g. from `your completion:6: error: Argument 1 to "Client" has incompatible type "str | None"; expected "str"  [arg-type] Found 1 error in 1 file (checked 1 source file)`extract the name "Client".

                        # Define the patterns
                        patterns = {
                            "has": re.compile(r'\"(.+?)\" has'),
                            "gets": re.compile(r'\"(.+?)\" gets'),
                            "for": re.compile(r'for \"(.+?)\"')
                        }

                        patterns_attribute = {
                            "has_attribute": re.compile(r'to \"(.+?)\" of \"(.+?)\" has'),
                            "gets_attribute": re.compile(r'\"(.+?)\" of \"(.+?)\" gets'),
                            "for_attribute": re.compile(r'\"(.+?)\" of \"(.+?)\" for'),
                            "missing_arg": re.compile(r'in call to \"(.+?)\" of \"(.+?)\"'),
                            "too_many_args": re.compile(r'for \"(.+?)\" of \"(.+?)\"'),
                            "signature_incompat": re.compile(r'Signature of \"(.+?)\" incompatible with supertype \"(.+?)\"')
                        }
                        # Find all matches for each pattern
                        matches = {key: pattern.findall(line) for key, pattern in patterns.items()}
                        matches = {k: v for k, v in matches.items()}
                        matches_attribute = {key: pattern.findall(line) for key, pattern in patterns_attribute.items()}
                        # Flatten the matches
                        matches_list = [match for k, v in matches.items() for match in v if '" of "' not in match]
                        matches_attribute_list = [f'{match[1]}.{match[0]}' for k, v in matches_attribute.items() for match in v]

                        # Combine the matches
                        all_matches = matches_list + matches_attribute_list

                        # Remove duplicates
                        all_matches = list(set(all_matches))

                        # Add the matches to the query
                        for match in all_matches:
                            if self.verbose:
                                print(f"[Signatures] Found problematic function or class: {match}")

                            # Sometimes, the queries are "type[ElasticsearchStore]" instead of "ElasticsearchStore".
                            # Extract the class name from the query.
                            if match.startswith("type["):
                                match = match[5:-1]

                            query.append(match)

        # Remove duplicates
        query = list(set([q for q in query if q.strip() != ""]))

        # If there is no query, there is nothing to do
        if len(query) == 0:
            if self.verbose:
                print("[Signatures] No problematic functions or classes found in the context.")
            os.remove(temp_file_name)
            return {
                "pass": True,
                "type": "info",
                "score": 0,
                "message": ""  # No message
            }

        # If there is a query, get the signatures and documentations of the functions and classes that match the query
        else:
            result = self.get_signature_and_doc(temp_file_name, list(set(query)))

            # Truncate the documentation to the first line (i.e. the signature)
            # Otherwise, the message will be too long
            for r in result:
                if r['doc']:
                    r['doc'] = r['doc'].split("\n")[0]

            if self.verbose:
                print("[Signatures] Got signatures and documentations:")
                for r in result:
                    print(f"[Signatures] {r['name']}: {r['signature']}, {r['doc']}")

            # If the analyzer could not find any signatures, return an error message
            if len(result) == 0:
                os.remove(temp_file_name)
                return {
                    "pass": False,
                    "type": "info",
                    "score": 0,
                    "message": "Cannot find the relevant signatures of " + ", ".join(query)
                }

            # Construct the feedback message
            result_str = "To fix these errors, use these ground truth signatures as a reference for your next completion:\n"
            result_str += "\n".join([f"{r['name']}: {r['signature'] if r['signature'] else r['doc']}" for r in result])

            # Clean up the temporary file
            os.remove(temp_file_name)

            # Return the result
            return {
                "pass": True,
                "type": "info",
                "score": - len(result),  # The more errors, the lower the score
                "message": result_str
            }
