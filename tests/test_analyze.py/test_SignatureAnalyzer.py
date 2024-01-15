# Generated with GPT-4 under supervision

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from llmcoder.analyze.SignatureAnalyzer import Import, SignatureAnalyzer


class TestSignatureAnalyzer(unittest.TestCase):

    def setUp(self) -> None:
        self.analyzer = SignatureAnalyzer()

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="import os\nimport sys\n")
    def test_get_imports(self, mock_file: MagicMock) -> None:
        expected_imports = [Import(['os'], None, 'os'), Import(['sys'], None, 'sys')]
        result = list(self.analyzer.get_imports("dummy_path"))
        self.assertEqual(result, expected_imports)

    def test_find_function_calls(self) -> None:
        code = "print('Hello World')\nos.system('ls')\n"
        query = ['print', 'os.system']
        result = self.analyzer.find_function_calls(code, query)
        expected_result = [(None, 'print'), ('os', 'system')]
        self.assertEqual(result, expected_result)

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="from os import path\nimport numpy as np\n")
    def test_get_imports_advanced(self, mock_file: MagicMock) -> None:
        # Test with from...import and aliased imports
        expected_imports = [Import(['os'], 'path', 'path'), Import(['numpy'], None, 'np')]
        result = list(self.analyzer.get_imports("dummy_path"))
        self.assertEqual(result, expected_imports)

    def test_find_function_calls_advanced(self) -> None:
        code = "np.array([1, 2, 3])\nclass MyClass:\n    def method(self):\n        pass\nobj = MyClass()\nobj.method()"
        query = ['np.array', 'method']
        result = self.analyzer.find_function_calls(code, query)
        expected_result = [('np', 'array'), ('obj', 'method')]
        self.assertEqual(result, expected_result)

    def test_dynamic_import_and_function_retrieval(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            temp_file.write("def function(arg1, arg2):\n    pass\n")
            temp_file_path = temp_file.name

        # Assuming the temporary file is in a directory that's not a package,
        # we need to add its directory to sys.path to make it importable
        temp_dir = os.path.dirname(temp_file_path)
        if temp_dir not in sys.path:
            sys.path.append(temp_dir)

        module_name = os.path.basename(temp_file_path).replace('.py', '')
        imported_module = __import__(module_name)

        # Try to access the function
        func = getattr(imported_module, "function", None)
        self.assertIsNotNone(func, "Function 'function' was not found in the imported module")

        # Clean up: remove the temp file and its directory from sys.path
        os.remove(temp_file_path)
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)
