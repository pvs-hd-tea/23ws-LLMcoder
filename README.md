<h1 align="center" style="margin-top: 0px;">LLMcoder: Feedback-Based Code Assistant</h1>
<h2 align="center" style="margin-top: 0px;">AI Methods and Tools for Programming</h2>

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

```sh
llmcoder [command] [options]
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
@software{llmcoder-hd-23,
    author = {Ana Carsi and Kushal Gaywala and Paul Saegert},
    title = {LLMcoder: Feedback-Based Code Assistant},
    month = mar,
    year = 2024,
    publisher = {GitHub},
    version = {0.0.1},
    url = {https://github.com/pvs-hd-tea/23ws-LLMcoder}
}
```
