import os
import re
import subprocess
import tempfile

from llmcoder.analyze.Analyzer import Analyzer


class MypyAnalyzer(Analyzer):
    """
    Analyzer that runs mypy on the code with the completion and returns the result.

    Parameters
    ----------
    input : str, optional
        The input code, by default "".
    completion : str, optional
        The completion to analyze, by default "".
    path : str, optional
        The path to the file to analyze, by default None.
    install_stubs : bool, optional
        Whether to install missing stubs, by default False.
    verbose : bool, optional
        Whether to print additional information, by default False.
    """
    def __init__(self, install_stubs: bool = False, verbose: bool = False):
        super().__init__(verbose=verbose)

        # Create a new temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            self.temp_file_name = temp_file.name

        self.install_stubs = install_stubs

    def __del__(self) -> None:
        """
        Removes the temp file on deletion.
        """
        if self.verbose:
            print("[Mypy] Cleaning up...")
        os.remove(self.temp_file_name)

    def setup(self, input: str) -> None:
        """
        Setup the analyzer with the input string for caching.

        Parameters
        ----------
        input : str
            Input string to be analyzed. Usually the user code since it is available at the time of initialization.
        """

        if self.verbose:
            print("[Mypy] Creating cache...")

        # Write the code to the temp file
        with open(self.temp_file_name, "w") as temp_file:
            # Only write import statements to the temp file
            # FIXME: This ignores multi-line inputs
            for line in input.split("\n"):
                if line.startswith("import ") or line.startswith("from "):
                    temp_file.write(line + "\n")

        # Run mypy so that the cache is created
        mypy_run = subprocess.run(["mypy", self.temp_file_name], capture_output=True, text=True)

        # Check if mypy reported missing stubs
        indicators = ["install-types"]
        missing_stubs = any(indicator in mypy_run.stdout or indicator in mypy_run.stderr for indicator in indicators)

        # Install missing stubs if requested
        if self.install_stubs and missing_stubs:
            if self.verbose:
                print("[Mypy] Installing missing stubs...")

                # Install missing stubs
                subprocess.run(["mypy", "--install-types", "--non-interactive"], capture_output=True, text=True)
        else:
            if self.verbose:
                print("[Mypy] No missing stubs found.")

    def analyze(self,
                input: str,
                completion: str,
                install_stubs: bool = True,
                mypy_args: list[str] | None = None,
                context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
        """
        Analyzes the completion using mypy.

        Parameters
        ----------
        input : str
            The input code.
        completion : str
            The completion to analyze.
        install_stubs : bool, optional
            Whether to install missing stubs, by default True.
        mypy_args : list[str], optional
            Additional arguments to pass to mypy, by default None.
        context : dict[str, dict[str, float | int | str]], optional
            The context of the completion, by default None.

        Returns
        -------
        dict
            The analysis result.
        """
        # Combine the input code and the completion

        with open(self.temp_file_name, "w") as temp_file:
            temp_file.write(input + completion)

        if mypy_args is None:
            # mypy_args = ["--disable-error-code=import-untyped"]  #FiXME: Should resolve error: Skipping analyzing "sklearn.ensemble": module is installed, but missing library stubs or py.typed marker  [import-untyped] but doesn't
            mypy_args = []

        # First run of mypy to check for missing stubs
        mypy_run = subprocess.run(["mypy", self.temp_file_name, *mypy_args], capture_output=True, text=True)

        # Check if mypy reported missing stubs
        indicators = ["install-types"]
        missing_stubs = any(indicator in mypy_run.stdout or indicator in mypy_run.stderr for indicator in indicators)

        # Install stubs if missing
        if install_stubs and missing_stubs:
            if self.verbose:
                print("[Mypy] Installing missing stubs...")

            # Install missing stubs
            subprocess.run(["mypy", "--install-types", "--non-interactive", *mypy_args], capture_output=True, text=True)

            # Re-run mypy after installing stubs
            mypy_run = subprocess.run(["mypy", self.temp_file_name, *mypy_args], capture_output=True, text=True)
        else:
            if self.verbose:
                print("[Mypy] No missing stubs found.")

        result = mypy_run.stdout if mypy_run.stdout else mypy_run.stderr

        # Remove all colors from the mypy output
        result = re.sub(r"\x1b\[[0-9;]*m", "", result)

        for line in result.split("\n"):
            if line.strip() != "":
                if self.verbose:
                    print(f"[Mypy] {line}")

        # Get the number of lines of the input code
        n_input_lines = len(input.split("\n")) - 1

        # Parse the line number from the mypy error message and filter the result
        filtered_result = []
        for line in result.split("\n"):
            if line.startswith(self.temp_file_name):
                line_number = int(line.split(":")[1])
                if line_number > n_input_lines:
                    filtered_result.append(line)
            else:
                filtered_result.append(line)

        # Replace the temp file name with "your completion". This helps the LLM understand that the error is caused by its completion.
        filtered_result = [line.replace(self.temp_file_name, "your completion") for line in filtered_result]

        # Remove the error message "your completion:2: error: Skipping analyzing." since it cannot be resolved by installing stubs
        filtered_result = [line for line in filtered_result if not re.match(r"your completion:\d+: error: Skipping analyzing", line)]

        # Remove the error message "your completion:16: note: See https:" since it does not provide any useful information
        filtered_result = [line for line in filtered_result if not re.match(r"your completion:\d+: note: See https:", line)]

        # Remove empty lines
        filtered_result = [line for line in filtered_result if line.strip() != ""][:10]

        # Construct the feedback string from the filtered result
        if len(filtered_result) == 0:
            filtered_result_str = "No mypy errors found."
        else:
            filtered_result_str = "The completion you provided resulted in the following errors:\n"
            filtered_result_str += "\n".join(filtered_result)

        # If there is no error in the mypy output, the completion passes the analysis
        passed = "error:" not in filtered_result_str

        # Return the result
        return {
            "type": "critical",
            "score": - len(re.findall(r"your completion:\d+: error:", filtered_result_str)),  # The more errors, the lower the score
            "pass": passed,
            "message": filtered_result_str if filtered_result_str else "No mypy errors found."
        }


# class DaemonMypyAnalyzer(Analyzer):
#     """
#     Analyzer that runs mypy on the code with the completion and returns the result.

#     Parameters
#     ----------
#     input : str, optional
#         The input code, by default "".
#     completion : str, optional
#         The completion to analyze, by default "".
#     path : str, optional
#         The path to the file to analyze, by default None.
#     install_stubs : bool, optional
#         Whether to install missing stubs, by default False.
#     verbose : bool, optional
#         Whether to print additional information, by default False.
#     """

#     def __init__(self, input: str = "", completion: str = "", path: str | None = None, install_stubs: bool = False, verbose: bool = False):
#         super().__init__(input=input, completion=completion, verbose=verbose)

#         if path is None:
#             # Create a new temp file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
#                 self.temp_file_name = temp_file.name
#                 temp_file.write(self.input + self.completion)
#         else:
#             self.temp_file_name = path

#         self.install_stubs = install_stubs

#     def __del__(self):
#         if self.verbose:
#             print("[Mypy] Cleaning up...")
#         os.remove(self.temp_file_name)
#         subprocess.run(["dmypy", "stop"], capture_output=True, text=True)

#     def _install_missing_stubs(self, mypy_output: str) -> None:
#         # Check if mypy reported missing stubs
#         indicators = ["install-types"]
#         missing_stubs = any(indicator in mypy_output for indicator in indicators)

#         # Install stubs if missing
#         if missing_stubs:
#             if self.verbose:
#                 print("[Mypy] Installing missing stubs...")

#             # Install missing stubs (This requires mypy, not the daemon!)
#             subprocess.run(["mypy", "--install-types", "--non-interactive"], capture_output=True, text=True)
#         else:
#             if self.verbose:
#                 print("[Mypy] No missing stubs found.")

#     def setup(self, input: str) -> None:
#         # Write the code to the temp file
#         with open(self.temp_file_name, "w") as temp_file:
#             temp_file.write(input)

#         # Run mypy so that the cache is created
#         if self.verbose:
#             print("[Mypy] Creating cache...")
#         mypy_run = subprocess.run(["dmypy", "run", self.temp_file_name], capture_output=True, text=True)

#         if self.install_stubs:
#             self._install_missing_stubs(mypy_run.stdout + mypy_run.stderr)

#     def analyze(self,
#                 input: str,
#                 completion: str,
#                 install_stubs: bool = False,
#                 mypy_args: list[str] | None = None,
#                 context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
#         """
#         Analyzes the completion using mypy.

#         Parameters
#         ----------
#         input : str
#             The input code.
#         completion : str
#             The completion to analyze.
#         install_stubs : bool, optional
#             Whether to install missing stubs, by default False.
#         mypy_args : list[str], optional
#             Additional arguments to pass to mypy, by default None.
#         context : dict[str, dict[str, float | int | str]], optional
#             The context of the completion, by default None.

#         Returns
#         -------
#         dict
#             The analysis result.
#         """
#         # Combine the input code and the completion
#         self.input = input
#         self.completion = completion

#         # Write the code to the temp file
#         with open(self.temp_file_name, "w") as temp_file:
#             temp_file.write(self.input + self.completion)

#         if mypy_args is None:
#             # mypy_args = ["--disable-error-code=import-untyped"]  #FiXME: Should resolve error: Skipping analyzing "sklearn.ensemble": module is installed, but missing library stubs or py.typed marker  [import-untyped] but doesn't
#             mypy_args = []

#         # First run of mypy to check for missing stubs
#         mypy_run = subprocess.run(["dmypy", "check", self.temp_file_name, *mypy_args], capture_output=True, text=True)

#         if install_stubs:
#             self._install_missing_stubs(mypy_run.stdout + mypy_run.stderr)

#             # Re-run mypy after installing stubs
#             mypy_run = subprocess.run(["dmypy", "check", self.temp_file_name, *mypy_args], capture_output=True, text=True)

#         result = mypy_run.stdout if mypy_run.stdout else mypy_run.stderr

#         # Remove all colors from the mypy output
#         result = re.sub(r"\x1b\[[0-9;]*m", "", result)

#         for line in result.split("\n"):
#             if line.strip() != "":
#                 if self.verbose:
#                     print(f"[Mypy] {line}")

#         # Get the number of lines of the input code
#         n_input_lines = len(input.split("\n")) - 1

#         # Parse the line number from the mypy error message and filter the result
#         filtered_result = []
#         for line in result.split("\n"):
#             if line.startswith(self.temp_file_name):
#                 line_number = int(line.split(":")[1])
#                 if line_number > n_input_lines:
#                     filtered_result.append(line)
#             else:
#                 filtered_result.append(line)

#         # Replace the temp file name with "your completion". This helps the LLM understand that the error is caused by its completion.
#         filtered_result = [line.replace(self.temp_file_name, "your completion") for line in filtered_result]

#         # Remove the error message "your completion:2: error: Skipping analyzing." since it cannot be resolved by installing stubs
#         filtered_result = [line for line in filtered_result if not re.match(r"your completion:\d+: error: Skipping analyzing", line)]

#         # Remove the error message "your completion:16: note: See https:" since it does not provide any useful information
#         filtered_result = [line for line in filtered_result if not re.match(r"your completion:\d+: note: See https:", line)]

#         # Remove empty lines
#         filtered_result = [line for line in filtered_result if line.strip() != ""]

#         # Construct the feedback string from the filtered result
#         if len(filtered_result) == 0:
#             filtered_result_str = "No mypy errors found."
#         else:
#             filtered_result_str = "The completion you provided resulted in the following errors:\n"
#             filtered_result_str += "\n".join(filtered_result)

#         # If there is no error in the mypy output, the completion passes the analysis
#         passed = "error:" not in filtered_result_str

#         # Return the result
#         return {
#             "type": "critical",
#             "score": - len(re.findall(r"your completion:\d+: error:", filtered_result_str)),  # The more errors, the lower the score
#             "pass": passed,
#             "message": filtered_result_str if filtered_result_str else "No mypy errors found."
#         }
