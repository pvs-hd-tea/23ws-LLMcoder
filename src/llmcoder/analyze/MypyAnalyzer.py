import os
import re
import subprocess
import tempfile

from llmcoder.analyze.Analyzer import Analyzer


class MypyAnalyzer(Analyzer):
    def analyze(self, input: str, completion: str, install_stubs: bool = True, mypy_args: list[str] | None = None, context: dict[str, dict[str, float | int | str]] | None = None) -> dict:
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

        Returns
        -------
        dict
            The analysis result.
        """

        code = input + completion

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code)

        if mypy_args is None:
            # mypy_args = ["--disable-error-code=import-untyped"]  #FiXME: Should resolve error: Skipping analyzing "sklearn.ensemble": module is installed, but missing library stubs or py.typed marker  [import-untyped] but doesn't
            mypy_args = []

        # First run of mypy to check for missing stubs
        mypy_run = subprocess.run(["mypy", temp_file_name, *mypy_args], capture_output=True, text=True)

        # Check if mypy reported missing stubs
        indicators = ["install-types"]
        missing_stubs = any(indicator in mypy_run.stdout or indicator in mypy_run.stderr for indicator in indicators)

        # Install stubs if missing
        if install_stubs and missing_stubs:
            print("[Mypy] Installing missing stubs...")

            # Install missing stubs
            subprocess.run(["mypy", "--install-types", "--non-interactive", *mypy_args], capture_output=True, text=True)

            # Re-run mypy after installing stubs
            mypy_run = subprocess.run(["mypy", temp_file_name, *mypy_args], capture_output=True, text=True)
        else:
            print("[Mypy] No missing stubs found.")

        result = mypy_run.stdout if mypy_run.stdout else mypy_run.stderr

        # Remove all colors from the mypy output
        result = re.sub(r"\x1b\[[0-9;]*m", "", result)

        for line in result.split("\n"):
            if line.strip() != "":
                print(f"[Mypy] {line}")

        # Get the number of lines of the input code
        n_input_lines = len(input.split("\n")) - 1

        # Parse the line number from the mypy error message and filter the result
        filtered_result = []
        for line in result.split("\n"):
            if line.startswith(temp_file_name):
                line_number = int(line.split(":")[1])
                if line_number > n_input_lines:
                    filtered_result.append(line)
            else:
                filtered_result.append(line)

        # Replace the temp file name with "your completion". This helps the LLM understand that the error is caused by its completion.
        filtered_result = [line.replace(temp_file_name, "your completion") for line in filtered_result]

        # Remove the error message "your completion:2: error: Skipping analyzing." since it cannot be resolved by installing stubs
        filtered_result = [line for line in filtered_result if not re.match(r"your completion:\d+: error: Skipping analyzing", line)]

        # Remove the error message "your completion:16: note: See https:" since it does not provide any useful information
        filtered_result = [line for line in filtered_result if not re.match(r"your completion:\d+: note: See https:", line)]

        if len(filtered_result) == 0:
            filtered_result_str = "No mypy errors found."
        else:
            filtered_result_str = "The completion you provided resulted in the following errors:\n"
            filtered_result_str += "\n".join(filtered_result)

        passed = "error:" not in filtered_result_str

        os.remove(temp_file_name)

        return {
            "type": "critical",
            "score": - len(re.findall(r"your completion:\d+: error:", filtered_result_str)),  # The more errors, the lower the score
            "pass": passed,
            "message": filtered_result_str if filtered_result_str else "No mypy errors found."
        }
