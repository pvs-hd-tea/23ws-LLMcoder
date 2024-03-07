# LLMcoder Documentation & Developer Guide

## Algorithm

## Project Structure

We follow common Python project structure conventions which includes

- `src/llmcoder/` - The main package containing the source code of the LLMcoder
- `src/llmcoder/__main__.py` - The entry point of the CLI
- `experimental/` - A directory for pilot tests, experiments, troubleshooting, etc. This directory mostly contains Jupyter notebooks.
- `configs/` - Configuration files used for evaluation of the LLMcoder
- `demos/` - Demo notebooks showing how to use the LLMcoder package

## `LLMCoder` class

### Parameters

- `analyzers` (list[str], optional): A list of analyzer names to use for evaluating code completions. Defaults to an empty list.
- `model_first` (str, optional): The model identifier for the first completion attempt. Defaults to `"ft:gpt-3.5-turbo-1106:personal::8LCi9Q0d"`.
- `model_feedback` (str, optional): The model identifier used for the feedback loop after the first completion. Defaults to the same as `model_first`.
- `feedback_variant` (str, optional): The feedback mechanism variant, either `"separate"` or `"coworker"`. Defaults to `"coworker"`.
- `system_prompt` (str | None, optional): A predefined system prompt to use. If `None`, a default system prompt based on preprocessing and fine-tuning is used. Defaults to `None`.
- `max_iter` (int, optional): The maximum number of iterations for running the feedback loop. Defaults to `10`.
- `backtracking` (bool, optional): Whether to enable backtracking during the feedback loop. Defaults to `True`.
- `log_conversation` (bool, optional): Enables logging of the conversation to a file. Defaults to `True`.
- `n_procs` (int, optional): The number of processes to use for running the analyzers in parallel. Defaults to `1`.
- `verbose` (bool, optional): Controls the verbosity of the output during execution. Defaults to `True`.

### Methods
---

```python
complete(code: str, temperature: float = 0.7, meta_temperature: float = 0.0, n: int = 1) -> str
```
Initiates the code completion process using the LLMCoder feedback loop, aiming to generate and iteratively refine code completions based on the provided code snippet. This method is the primary interface for users to interact with the LLMCoder system for code completion tasks.

**Parameters**:
- `code` (str): The code snippet that needs completion or enhancement.
- `temperature` (float, optional): Controls the randomness of the completion. Higher values result in more varied outputs. Defaults to `0.7`.
- `meta_temperature` (float, optional): Adjusts the selection criteria among different conversation paths in the feedback loop, influencing the exploration of completion options. Defaults to `0.0`.
- `n` (int, optional): The number of completion alternatives to generate at each feedback loop iteration. Defaults to `1`.

**Returns**: A string representing the best code completion after iterating through the feedback loop.

---

```python
_create_conversation_file() -> str
```
Creates a file to log the conversation history, including code completions and interactions with the analyzers. This method facilitates debugging and analysis of the LLMCoder's performance.

**Parameters**: None

**Returns**: A string representing the file path to the newly created conversation log file.

---

```python
_get_best_completion(conversations: list[Conversation]) -> str:
```
Identifies and retrieves the best code completion from a list of conversations based on their scores. This method is critical for selecting the most promising code completion after the feedback loop.

**Parameters**:
- `conversations` (list[Conversation]): A list of `Conversation` objects containing potential code completions and their associated scores.

**Returns**: A string representing the best code completion from the provided conversations.

---

```python
_reset_loop()
```
Resets the feedback loop and internal variables to their initial states. This includes resetting iteration counters, token generation counts, and initializing the priority queue for conversations. Essential for starting a fresh code completion process.

**Parameters**: None

**Functionality**: Resets internal state without returning any value.

---

```python
_get_completions_for(self, conversation: Conversation, model: str = 'gpt-3.5-turbo', temperature: float = 0.7, n: int = 1, max_retries: int = 5, delta_temperature: float = 0.2, max_temperature: float = 2, factor_n: int = 2, max_n: int = 32) -> None
```
The `_get_completions_for` method is a crucial component of the `LLMCoder` class, responsible for generating and evaluating code completions based on a given conversation context. This method interacts with OpenAI's API to fetch code completions, filters out unsuitable options based on previous mistakes, and employs analyzers to evaluate the quality of the completions. It aims to refine the selection process through iterative adjustments and parallel processing, ultimately contributing to the enhancement of code generation.

#### Parameters

- `conversation` (Conversation): Represents the current state of the conversation, including all messages exchanged so far. This object is used to maintain context for generating relevant code completions.

- `model` (str, optional): Specifies the OpenAI model identifier to use for generating completions. The default model is `'gpt-3.5-turbo'`, known for its efficiency and effectiveness in a wide range of code-related tasks.

- `temperature` (float, optional): A parameter that controls the randomness of the generated completions. Higher values result in more diverse outputs, while lower values tend to produce more deterministic and conservative completions. The default value is `0.7`.

- `n` (int, optional): Determines the number of completion options to generate for each request to the OpenAI API. A higher number provides a broader range of choices for analysis but requires more computational resources. The default is `1`.

- `max_retries` (int, optional): The maximum number of attempts to generate a valid completion before giving up. This parameter helps in dealing with repeated mistakes or unsuitable completions by allowing the method to try different configurations. The default value is `5`.

- `delta_temperature` (float, optional): Specifies the amount by which the temperature parameter should be increased after each retry when the generated completions are deemed unsuitable. Increasing the temperature helps in diversifying the outcomes in search of a valid completion. The default increment is `0.2`.

- `max_temperature` (float, optional): The upper limit for the temperature parameter, ensuring that the method does not produce overly random or irrelevant completions. The default maximum temperature is set to `2`.

- `factor_n` (int, optional): Defines the multiplication factor for increasing the number of completion options (`n`) after each retry. This parameter allows the method to explore a wider range of completions in case of repeated failures. The default factor is `2`.

- `max_n` (int, optional): The maximum allowable number of completion options to generate in a single request. This limit prevents excessive computational load and API usage. The default maximum is `32`.

#### Functionality

The method begins by requesting a set of completions from OpenAI's API using the initial parameters provided. It then proceeds to filter these completions by removing any that replicate previous mistakes or are duplicates. If the resulting set of completions is insufficient (due to filtering or other factors), the method adjusts the temperature and `n` parameters to generate a new set of options, repeating this process up to `max_retries` times if necessary.

For each valid completion, the method employs the configured analyzers to evaluate and score the quality and relevance of the completion in parallel. This evaluation process helps in identifying the most promising completions based on analytical criteria.

The outcomes of the analyzers are then used to update the conversation object with new messages and analysis results, effectively pushing updated conversations onto a priority queue for further consideration in the feedback loop.

This method does not return a value directly; instead, it significantly influences the state of the `LLMCoder`'s internal priority queue by adding new, analyzed conversation objects. These objects contain updated completions along with their evaluations, ready for further processing in the feedback loop to refine the code generation process.

---

```python
_run_analyzers(code: str, completion: str) -> dict[str, dict]
```
Runs the configured analyzers on a given code snippet and its completion. This method evaluates the quality and relevance of the completion, contributing to the scoring mechanism that guides the feedback loop.

**Parameters**:
- `code` (str): The original code snippet provided for completion.
- `completion` (str): The generated code completion to be analyzed.

**Returns**: A dictionary containing the analysis results from each configured analyzer.

---

```python
_feedback_prompt_template(result_messages: list[str]) -> str
```
Constructs a feedback prompt based on the messages resulting from previous analyses. This method formats the feedback for inclusion in the next iteration of code completion requests, guiding the model towards improved outputs.

**Parameters**:
- `result_messages` (list[str]): A list of messages generated by analyzers, encapsulating feedback on prior completions.

**Returns**: A string representing the constructed feedback prompt.

---

```python
_step(code: str, temperature: float = 0.7, meta_temperature: float = 0.0, n: int = 1)
```
Executes a single iteration of the feedback loop for code completion. This involves selecting a conversation based on scores, generating new completions based on feedback, and updating the conversation queue with these new completions.

**Parameters**:
- `code` (str): The code snippet for which completions are being generated.
- `temperature` (float, optional): Controls the randomness of completions.
- `meta_temperature` (float, optional): Influences the selection of conversation paths for exploration.
- `n` (int, optional): The number of completions to generate in this step.


### Evaluation

## Future Work