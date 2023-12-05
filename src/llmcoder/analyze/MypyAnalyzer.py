import subprocess
import tempfile
import os

from llmcoder.analyze.Analyzer import Analyzer


class MypyAnalyzer(Analyzer):
    def analyze(self, input: str, completion: str, install_stubs=True) -> dict:
        code = input + completion

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code)

        # First run of mypy to check for missing stubs
        first_run = subprocess.run(["mypy", temp_file_name], capture_output=True, text=True)
        if install_stubs and ("install-types" in first_run.stderr or "install-types" in first_run.stdout):
            print("Installing missing stubs...")
            # Install missing stubs
            subprocess.run(["mypy", "--install-types", "--non-interactive"], capture_output=True, text=True)
            # Re-run mypy after installing stubs
            second_run = subprocess.run(["mypy", temp_file_name], capture_output=True, text=True)
            result = second_run.stdout if second_run.stdout else second_run.stderr
        else:
            print("No missing stubs found.")
            result = first_run.stdout if first_run.stdout else first_run.stderr

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

        filtered_result = "\n".join(filtered_result).replace(temp_file_name, "your completion")

        os.remove(temp_file_name)
        return {
            "pass": "error:" not in filtered_result,
            "message": filtered_result if filtered_result else "No mypy errors found."
        }
