import subprocess
import tempfile
import os

from llmcoder.analyze.Analyzer import Analyzer


class MypyAnalyzer(Analyzer):
    def analyze(self, code: str, install_stubs=False) -> dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code)

        # First run of mypy to check for missing stubs
        first_run = subprocess.run(["mypy", temp_file_name], capture_output=True, text=True)
        if install_stubs and ("install-types" in first_run.stderr or "install-types" in first_run.stdout):
            print("Installing missing stubs...")
            # Install missing stubs
            subprocess.run(["mypy", "--install-types", "--non-interactive", "--strict"], capture_output=True, text=True)
            # Re-run mypy after installing stubs
            second_run = subprocess.run(["mypy", temp_file_name], capture_output=True, text=True)
            result = second_run.stdout if second_run.stdout else second_run.stderr
        else:
            print("No missing stubs found.")
            result = first_run.stdout if first_run.stdout else first_run.stderr

        os.remove(temp_file_name)
        return result
