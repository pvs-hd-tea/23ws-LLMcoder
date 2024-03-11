import unittest

from llmcoder.analyze.mypy_analyzer import MypyAnalyzer
from llmcoder.analyze.signature_analyzer import SignatureAnalyzer


class TestSignatureAnalyzerContext(unittest.TestCase):

    def test_has_incompatible_type(self) -> None:
        # Argument 1 to "analyze" of "APIDocumentationAnalyzer" has incompatible type "Callable[[object], str]"; expected "str"  [arg-type]
        code = """
from llmcoder.analyze import APIDocumentationAnalyzer
a = APIDocumentationAnalyzer()
analyzer_results = [a.an"""

        completion = """alyze(input, completion, generator)]"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("APIDocumentationAnalyzer.analyze" in sig_result['message'])

    def test_gets_multiple_values(self) -> None:
        # "analyze" of "APIDocumentationAnalyzer" gets multiple values for keyword argument "input"  [misc]
        code = """
from llmcoder.analyze import APIDocumentationAnalyzer
a = APIDocumentationAnalyzer()
analyzer_results = [a.an"""

        completion = """alyze(input="a", input="b", completion="v")]"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("APIDocumentationAnalyzer.analyze" in sig_result['message'])

    def test_unexpected_keyword_argument(self) -> None:
        # Unexpected keyword argument "seed" for "LLMCoder"  [call-arg]
        code = """
from llmcoder import LLMCoder
coder = LLMCoder("""

        completion = """seed=1)"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("LLMCoder: (" in sig_result['message'])

    def test_too_many_arguments(self) -> None:
        # Too many arguments for "APIDocumentationAnalyzer"  [call-arg]
        code = """
from llmcoder.analyze import APIDocumentationAnalyzer
a = APIDocumentationAnalyzer("""

        completion = """1, 2, 3, 4, 5, 6, 7)"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("APIDocumentationAnalyzer: (" in sig_result['message'])

    def test_attribute_has_incompatible_type(self) -> None:
        # Argument 1 to "analyze" of "APIDocumentationAnalyzer" has incompatible type "int"; expected "str"  [arg-type]
        code = """
from llmcoder.analyze import APIDocumentationAnalyzer
a = APIDocumentationAnalyzer()

a.analyze("""

        completion = """1, completion="a")"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("APIDocumentationAnalyzer.analyze" in sig_result['message'])

    def test_attribute_gets_multiple_values(self) -> None:
        # analyze" of "APIDocumentationAnalyzer" gets multiple values for keyword argument "input"  [misc]
        code = """
from llmcoder.analyze import APIDocumentationAnalyzer
a = APIDocumentationAnalyzer()

a.analyze("""

        completion = """input="a", input="b", completion="a")"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("APIDocumentationAnalyzer.analyze" in sig_result['message'])

    def test_attribute_missing_positional_arguments(self) -> None:
        # Missing positional arguments "input", "completion" in call to "analyze" of "APIDocumentationAnalyzer"  [call-arg]
        code = """
from llmcoder.analyze import APIDocumentationAnalyzer
a = APIDocumentationAnalyzer()
analyzer_results = [a.an"""

        completion = """alyze()]"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("APIDocumentationAnalyzer.analyze" in sig_result['message'])

    def test_attribute_too_many_arguments(self) -> None:
        # Too many arguments for "analyze" of "APIDocumentationAnalyzer"  [call-arg]
        code = """
from llmcoder.analyze import APIDocumentationAnalyzer
a = APIDocumentationAnalyzer()
analyzer_results = [a.an"""

        completion = """alyze("a", "b", {"a": {"a": True}}, "d")]"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("APIDocumentationAnalyzer.analyze" in sig_result['message'])

    def test_attribute_signature_incompatible_with_supertype(self) -> None:
        # Signature of "analyze" incompatible with supertype "Analyzer"  [override]
        code = """
import os
import re
import subprocess
import tempfile

from llmcoder.analyze.analyzer import Analyzer

class MypyAnalyzer(Analyzer):
    def analyze("""

        completion = """self, input: str, completion: str, install_stubs: bool = True, *mypy_args: str | None) -> dict:
                return {}"""

        mypy = MypyAnalyzer(verbose=True)
        sig = SignatureAnalyzer(verbose=True)

        mypy_result = mypy.analyze(code, completion)
        sig_result = sig.analyze(code, completion, {'mypy_analyzer_v1': mypy_result})

        self.assertTrue("Analyzer.analyze" in sig_result['message'])
