import os
import sys
from abc import ABC, abstractmethod
from typing import Generator

import jedi
from jedi.api.classes import Completion, Name, Signature
from jedi.api.errors import SyntaxError


class ResponseTypes:
    List = list
    Generator = Generator
    Name = Name
    Signature = Signature
    Completion = Completion
    SyntaxError = SyntaxError


class AbstractResponse(ABC):
    @abstractmethod
    def print_response(self, completions: ResponseTypes.Name | ResponseTypes.Generator | ResponseTypes.List | ResponseTypes.Signature):
        pass


class NameType(AbstractResponse):
    def print_response(self, completions: ResponseTypes.Name):
        print(completions)
        print(f"completions.full_name: {completions.full_name}")
        print(f"completions.description: {completions.description}")


class GeneratorType(AbstractResponse):
    def print_response(self, completions: ResponseTypes.Generator):
        print(f"CompleteSearch Generator: {completion}")


class ListType(AbstractResponse):
    """ Completions / Responses Generator """
    def print_response(self, completions: ResponseTypes.List):
        if len(completions) == 0:
            return
        for completion in completions:
            yield completion


class SignatureType(AbstractResponse):
    def print_response(self, index, completion: ResponseTypes.Signature):
        print(f"Signatures[{index}]: {completion}")


curr_py_env = sys.path
project_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llmcoder", "23ws-LLMcoder")
file_path = os.path.join(project_dir, "src", "llmcoder", "analyze", "GPTReviewAnalyzer.py")
curr_source_code = ""
with open(file_path, "r") as file:
    curr_source_code = file.read()


source: str = """import json
json.load("""
source_lines: list[str] = source.splitlines()
source_line: int = 82
source_col: int = 42

jedi.preload_module(["os", "re", "openai", "llmcoder"])
project = jedi.Project(path=project_dir, environment_path="/home/kushal/.config/python.env/llmcoder.env")
script = jedi.Script(curr_source_code, path=file_path, project=project, environment=project.get_environment())
# completions = script.complete_search()
completions = script.infer(source_line, source_col)
# completions = script.get_signatures(source_line, source_col)
# completions = script.goto(source_line, source_col)
# completions = script.infer(source_line, source_col)
# completions = script.get_references(source_line, source_col)
# completions = script.complete(source_line, source_col)
# completions = script.get_context(source_line, source_col)
# completions = script.get_syntax_errors()

print("="*50)
print(f"type(completions): {type(completions)}")
print("-"*20)
if isinstance(completions, ResponseTypes.Name):
    # Get context
    print(f"Get context: {completions}")
    print("-"*20)
elif isinstance(completions, ResponseTypes.List):
    if len(completions) == 0:
        print("Empty Responses")
        print("-"*20)
    for index, completion in enumerate(completions):
        if isinstance(completion, ResponseTypes.Name):
            if hasattr(completion, completion.full_name):
                print(f"Yes it has!!")
            # Get references
            print(f"References[{index}]: {completion}")
            print(f"Completions[{index}].full_name: {completion.full_name}")
            print(f"Completions[{index}].description: {completion.description}")
            print("-"*20)
        elif isinstance(completion, ResponseTypes.Signature):
            # Signatures
            print(f"Signatures[{index}]: {completion.docstring()}")
            print("-"*20)
        elif isinstance(completion, ResponseTypes.Generator):
            # Complete search
            print(f"Complete Search[{index}]: {next(completion)}")
            print("-"*20)
        elif isinstance(completion, ResponseTypes.SyntaxError):
            # Syntax errors
            print(f"SyntaxErrors[{index}]: {completion}")
            print(f"Error line: {completion.line}")
            print(f"Error column: {completion.column}")
            print(f"Error until line: {completion.until_line}")
            print(f"Error until column: {completion.until_column}")
            print(f"Error message: {completion.get_message()}")
            print("-"*20)
        elif isinstance(completion, ResponseTypes.Completion):
            # Get completion suggestions
            print(f"Completions[{index}]: {completion}")
            print(f"Completions[{index}].complete: {completion.complete}")
            print(f"Completions[{index}].name: {completion.name}")
            print("-"*20)
        else:
            print(f"Completions[{index}]: {completion}")
            print(f"Completions[{index}].complete: {completion.complete}")
            print(f"Completions[{index}].name: {completion.name}")
            print("-"*20)

print(f"Script: {script}")
print("="*50)