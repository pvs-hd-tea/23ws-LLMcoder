<div align="center">

<img src="./images/LLMcoder_round_inline_black.png" width="200" alt="LLMcoder Logo" />

</div>

<h1 align="center" style="margin-top: 0px;"><b>LLMcoder</b> - A Feedback-Based Coding Assistant</h1>
<h2 align="center" style="margin-top: 0px;">Practical AI Methods and Tools for Programming</h2>

<div align="center">

[![Pytest](https://github.com/pvs-hd-tea/23ws-LLMcoder/actions/workflows/pytest.yml/badge.svg)](https://github.com/pvs-hd-tea/23ws-LLMcoder/actions/workflows/pytest.yml)
[![Code Quality](https://github.com/pvs-hd-tea/23ws-LLMcoder/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/pvs-hd-tea/23ws-LLMcoder/actions/workflows/pre-commit.yml)
</div>

# Introduction
# Requirements
## Hardware
## Software
- Python >= 3.10
- `pip` >= [21.3](https://pip.pypa.io/en/stable/news/#v21-3)
# Getting Started
## Clone The Repository

```sh
git clone https://github.com/pvs-hd-tea/23ws-LLMcoder
cd 23ws-LLMcoder
```

## Create A Virtual Environment (optional):

### With conda

```sh
conda create -n llmcoder python=3.11 [ipykernel]
conda activate llmcoder
```

## Install

### Install the package

```sh
pip install -e .
```

## Usage

CLI:
```sh
llmcoder [command] [options]
```

Python API:
```python
from llmcoder import LLMcoder

llmcoder = LLMcoder(
    analyzers=[
        "mypy_analyzer_v1",  # Detect and fix type errors
        "signature_analyzer_v1",  # Augment type errors with signatures
        "gpt_score_analyzer_v1"],  # Score and find best completion
    feedback_variant="coworker",  # Make context available to all analyzers
    max_iter=3,  # Maximum number of feedback iterations
    n_procs=4  # Complete and analyze in parallel
    verbose=True  # Print progress
)

# Define an incomplete code snippet
code = "print("

# Complete the code
result = llmcoder.complete(code, n=4)
```

# Development

## Setup
To set up the development environment, run the following commands:

```sh
pip install -e .[dev]
pre-commit install
```


# Citation
```bibtex
@software{llmcoder-hd-24,
    author = {Ana Carsi and Kushal Gaywala and Paul Saegert},
    title = {LLMcoder: Feedback-Based Code Assistant},
    month = mar,
    year = 2024,
    publisher = {GitHub},
    version = {0.2},
    url = {https://github.com/pvs-hd-tea/23ws-LLMcoder}
}
```
