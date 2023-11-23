import ast
import importlib
import re
import json

class APIDoc :
    def __init__(code: str):
        this.code = code
        this.packages = []
        this.documentations = []

    def extract_import_packages(code: str) -> list[dict[str, str]]: 
        """
        :param code: The code that is to be analyzed for the import packages
        :param code: str
        :return: returns a list of packages that is being fetched from the input
        :rtype: dict[]
        """

        response_tree = ast.parse(this.code)

        for node in ast.walk(response_tree):
            if "names" in dir(node):
                if "module" in dir(node):
                    this.packages.append({"module": node.module, "name": node.names[0].name})
                else:
                    this.packages.append({"name": node.names[0].name, "asname": node.names[0].asname})

        return packages

    def get_documentation_from_packages(packages: dict[str, str]) -> list[dict[str, str]]:
        """
        :param packages: list of packages
        :return: returns list of {"package": package, "documentation": documentation}
        :rtype: list[dict]dict[]
        """

        for package in this.packages:
            if "module" in package:
                documentation = this.get_from_documentation(package)
                this.documentation.append(documentation)
            else:
                documentation = this.get_import_documentation(package)
                this.documentation.append(documentation)
        
        return this.documentations

    def get_from_documentation(module: dict[str, str]) -> dict[str, str]:
        from_module = module["module"]
        module_name = module["name"]

        this.import_package(from_module)

        return

    def import_package(package: str):
        spec = importlib.util.find_spec(package)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    def get_import_documentation(package: dict[str, str]) -> dict[str, str]:
        return

class Analyzer(Observer):
    def analyze(self, code):
        raise NotImplementedError

class ExecutabilityTester(Analyzer):
    def analyze(self, code):
        # Perform analysis
        error = self.check_executability(code)
        if error:
            self.notify_observers(error)  # Notify other analyzers about the error

class APIDocumentationAnalyzer(Analyzer):
    def update(self, message):
        # Act upon receiving a notification
        if message.type == 'ExecutabilityError':
            documentation = self.fetch_documentation(message.details)
            # Process the documentation further for more targeted search

# Setting up the observers
exec_tester = ExecutabilityTester()
api_doc_analyzer = APIDocumentationAnalyzer()

exec_tester.add_observer(api_doc_analyzer)  # api_doc_analyzer now listens to exec_tester
class Combinator(Analyzer):
    def analyzer(self, FromAnalyzer, ToAnalyzer):
        
