import ast
import importlib.util
import pydoc
import re
from types import ModuleType
from typing import Any

from llmcoder.analyze.Analyzer import Analyzer


class APIDocumentationAnalyzer(Analyzer):
    def __init__(self) -> None:
        """
        APIDocumentationAnalyzer analyzes the code passed in the object and fetches the documentations of the API references

        Parameters
        ----------
        code : str
            This string represents the code that is to be analyzed by this analyzer
        """
        self.code: str = ""
        self.module: str = ""
        # List to save modules already imported
        self.modules: list[str] = list()
        # To keep track of spec state
        self.spec = None

    def analyze(self, input: str, completion: str) -> dict[str, bool | str]:
        """
        analyze analyzes the code that is passed in APIDocumentAnalyze class object and returns the documentation with the API References

        Returns
        -------
        documentations : str
            documentations along with corresponding packages that is being identified and fetched from the code
        """
        analysis_results = self._analyze(input, completion)

        # Convert the list of dictionaries to a string
        documentations = "\n".join(
            [
                f"{result['module']}\n{result['documentation']}"
                for result in analysis_results
            ]
        )

        return {
            "pass": False,  # FIXME: Maks sure that the functions are used correctly!
            "message": documentations,
        }

    # The implementation of the abstract method
    def _analyze(self, input: str, completion: str) -> list[dict[str, str]]:
        """
        analyze analyzes the code that is passed in APIDocumentAnalyze class object and returns the documentation with the API References

        Returns
        -------
        documentations : list[str]
            list documentations along with corresponding packages that is being identified and fetched from the code
        """
        self.input = input
        self.completion = completion

        self.code = self.input + self.completion

        # List to save all documentations
        documentations: list[dict[str, str]] = list()

        for node in ast.walk(ast.parse(self.code)):
            # Declerations to import the module and fetch the documentation of the module API
            # Check names in node attributes to get all the packages
            if hasattr(node, "names"):
                # Check if its a from statement import
                if isinstance(node, ast.ImportFrom):
                    # Save module name
                    # Save attributes / submodules names
                    node_module, module_names = self.get_node_module_names(node)
                    documentations.extend(
                        self.get_docs_from_from_statements(node_module, module_names)
                    )
                # If import is import statement
                else:
                    # Then, iterate over all the import <packages> i.e.: node.names
                    documentations.extend(
                        self.get_docs_from_import_statements(node.names)
                    )

        return documentations

    def get_docs_from_from_statements(
        self, node_module: str, module_names: list[str]
    ) -> list[dict[str, str]]:
        documentations: list[dict[str, str]] = []

        # Traverse all module attributes / submodules
        # attributes / submodules in the from statements
        for names in module_names:
            # attribute / submodules name
            module_name, module_asname = self.get_module_name_asname(names)

            if node_module is None:
                continue

            module = self.import_module(node_module)

            if not hasattr(module, module_name):
                print("Is not an Attribute")
                print(module_name, module_asname)

                attributes: list[str] = list()

                # Check if a alias is used for a submodule or for an attribute
                if module_asname is not None:
                    # Alias of submodules / attributes used in the code suggestion
                    asnames = self.get_asnames_from_module(module_asname)
                else:
                    # Alias of submodules / attributes used in the code suggestion
                    asnames = self.get_asnames_from_module(module_name)

                # Attributes used in the code suggestion
                # Submodules used in the code suggestion
                print(f"ASName: {asnames}")
                attributes, submodules = self.get_attributes_submodules_lists(asnames)

                # Check if submodules are used directly to access an attribute
                # Get docs from all the attributes of the associated module
                if len(submodules) > 0:
                    if len(submodules[0]) > 0:
                        print(submodules, attributes)
                        for attribute, submodule in zip(attributes, submodules):
                            print(submodule, attribute)
                            # Create a submodule with module to import module.submodule
                            module_submodule_name = ".".join([node_module, submodule])
                            # Import the module.submodule
                            module = self.import_module(module_submodule_name)
                            # Get the attribute that is associated with the module.submodule
                            attribute = getattr(module, attribute)
                            # Fetch the documentation of the module.submodule.attribute
                            documentations.append(
                                self.format_documentation(
                                    ".".join([module_submodule_name, attribute]),
                                    self.fetch_documentation(attribute)
                                )
                            )
                    else:
                        for attribute in attributes:
                            module_submodule_name = ".".join([node_module, module_name])
                            module = self.import_module(module_submodule_name)
                            attrib = getattr(module, attribute)
                            documentations.append(
                                self.format_documentation(
                                    ".".join([module_submodule_name, attribute]),
                                    self.fetch_documentation(attrib)
                                )
                            )
                else:
                    # Create the module with submodule module.submodule
                    module_submodule_name = ".".join([node_module, module_name])
                    # Import the module with submodule module.submodule
                    module = self.import_module(module_submodule_name)

                    # Traverse over all the attributes of the module.submodule
                    for attribute in attributes:
                        # Get the attribute that is associated with the module.submodule
                        attribute = getattr(module, attribute)
                        # Fetch the documentation of the module.submodule.attribute
                        documentations.append(
                            self.format_documentation(
                                ".".join([module_submodule_name, attribute]),
                                self.fetch_documentation(attribute)
                            )
                        )
            # If the the import after from is an attribute then directly get the documentation
            else:
                # No need to import the module as it was done before while checking hasattr
                # Get the attribute that is associated with the module
                print("Is an Submodule")
                print(node_module, module_name)
                attribute = getattr(module, module_name)
                # Fetch the documentation of the module.attribute
                documentations.append(
                    self.format_documentation(
                        ".".join([node_module, module_name]),
                        self.fetch_documentation(attribute)
                    )
                )

        return documentations

    def get_docs_from_import_statements(self, packages: Any) -> list[dict[str, str]]:
        documentations = []

        for package in packages:
            package_name: str = package.name
            package_asname: str = package.asname

            if package_asname is not None:
                asnames = self.get_asnames_from_module(package_asname)
            else:
                asnames = self.get_asnames_from_module(package_asname)
                asnames = [
                    ref.split(".")[1::1]
                    for ref in re.findall(
                        rf"{package_name}\.[a-zA-Z_\.0-9]+", self.code
                    )
                ]

            for asname in asnames:
                reference, submodule = self.get_references_submodules(asname)

                if len(submodule) != 0:
                    package_submodule_name = ".".join([package_name, submodule])
                    module = self.import_module(package_submodule_name)
                else:
                    package_submodule_name = package_name
                    module = self.import_module(package_name)

                try:
                    function = getattr(module, reference)

                    documentations.append(
                        self.format_documentation(
                            ".".join([package_submodule_name, reference]),
                            self.fetch_documentation(function)
                        )
                    )
                except AttributeError as e:
                    documentations.append(
                        self.format_documentation(
                            ".".join([package_submodule_name, reference]),
                            str(e)
                        )
                    )
        return documentations

    def get_attributes_submodules_lists(self, asnames: list[str]) -> tuple[list, list]:
        print(f"ASNames: \n{asnames}\n\n")
        return [asname[-1] for asname in asnames], [asname[:-1:1] for asname in asnames]

    def get_asnames_from_module(self, module_name: str) -> list[str]:
        return [
            reference.split(".")[1::1]
            for reference in re.findall(rf"{module_name}\.[a-zA-Z_\.0-9]+", self.code)
        ]

    def get_references_submodules(self, asname: str | list[str]) -> tuple[str, str]:
        return asname[-1], ".".join(asname[:-1:1])

    def fetch_documentation(self, attribute: Any) -> str:
        return pydoc.render_doc(attribute)

    def format_documentation(self, module: str, documentation: str) -> dict[str, str]:
        return {"module": module, "documentation": documentation}

    def get_node_module_names(self, node: Any) -> tuple[str, list[Any]]:
        return node.module, node.names

    def get_module_name_asname(self, names: Any) -> tuple[str, str]:
        return names.name, names.asname

    def import_module(self, package: str) -> ModuleType | str:
        spec: Any = importlib.util.find_spec(package)

        if spec is not None:
            module: ModuleType | str = importlib.util.module_from_spec(spec)
            if module is not None:
                spec.loader.exec_module(module)
                print(f"Module: \n{module}\n\n")
                return module

        return package
