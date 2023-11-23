import ast
import importlib
import re
import json
import sys

class APIDocumentationAnalyzer :
    def __init__():
        self.code = None
        self.spec = None
        self.module = None

    def analyze(self, code: str) -> list[str]:
        """
        :param code: The code that is to be analyzed for the import packages
        :param code: str
        :return: returns a list of packages that is being fetched from the input
        :rtype: list[str]
        """
        documentations: list[dict] = []
        
        response_tree = ast.parse(code)
        
        for node in ast.walk(response_tree):
            documentation: list[dict] = []
            if "names" in dir(node):
                if "module" in dir(node):
                    # from module import name
                    # Import the module
                    self.import_package_modules(node.module, node.names[0].name)
                    # Get the references of the module where it is used
                    references = self.get_reference_packages(node.names[0].name)
                    # Get the documentations of the references of the module
                    # documentations = self.get_documentation_from_packages(node.names[0].name, references)
                    documentation.append(self.get_documentation_from_packages(node.names[0].name, references))
                else:
                    if node.names[0].asname is None:
                        # import name
                        self.import_package_modules(node.names[0].name, reference_packages)
                        references = self.get_reference_packages(node.names[0].name)
                        # documentations = self.get_documentation_from_packages(node.names[0].name, references)
                        documentation.append(self.get_documentation_from_packages(node.names[0].name, references))
                    else:
                        # import name as asname
                        # Get references
                        self.import_package_modules(node.names[0].name, reference_packages)
                        references = self.get_reference_packages(node.names[0].asname)
                        documentation.append(self.get_documentation_from_packages(node.names[0].name, references))
            documentations.extend(documentation)
        return documentations

    def import_package_modules(self, packages: list[str]):
        """
        Imports the packages and modules that are needed to fetch the documentations
        :param packages: list of packages
        """

        if module in sys.modules:
            self.spec = importlib.util.find_spec(package)
            self.module = importlib.util.module_from_spec(self.spec)
        self.spec.loader.exec_module(module)        

    def get_reference_packages(self, packages: list[str], code) -> list[str]:
        """
        :param package: The package that is to be analyzed for the reference packages
        :param package: str
        :return: returns a list of packages that is being fetched from the input
        :rtype: list[str]
        """
        # Get the references of the package where it is used
        package = package.split(".")[1]
        # Get the documentations of the references of the package
        return [re.findall(fr"{package}\.[a-zA-Z_]+", code).split(".")[1] for package in packages]

    def get_documentation_from_packages(self, package: str, references: list[str]) -> list[dict[str, str]]:
        """
        :param packages: list of packages
        :return: returns list of {"package": package, "documentation": documentation}
        :rtype: list[dict]
        """
        # Import the function from the package then fetch docs
        return [self.create_package_dict(package, pydoc.render_doc(getattr(package, reference))) for reference in references]

    def create_package_dict(self, package: str, documentation: str) -> dict[str, str]:
        """
        :param package: package name
        :param documentation: documentation of the package
        :return: returns {"package": package, "documentation": documentation}
        :rtype: dict[str, str]
        """
        return {"package": package, "documentation": documentation}