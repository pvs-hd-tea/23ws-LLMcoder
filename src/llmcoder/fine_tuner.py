# Concrete Class implementing Fine Tuning
# TODO: coherence with already existing code

import json
import os
import numpy as np
# import tiktoken


class FineTuner():
    def __init__(self, repository_list: list) -> None:
        self.repository_list = repository_list

    def get_repos_with_query(self, query: str, num_repos: int):
        pass

    def get_popular_repos(self, num_repos_per_query: int):
        pass

    # TODO: try-cath exception handling 
    def clone_repository(self, repo_url: str, output_dir: str):
        pass

    # TODO: try-cath exception handling 
    def extract_python_files(self, repo_dir: str, output_dir: str):
        pass
    
    