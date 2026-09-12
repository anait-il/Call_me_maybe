*This project has been created as part of the 42 curriculum by anait-il.*

# Call Me Maybe

## Description

### Goal

**Call Me Maybe** is a project focused on understanding how Large Language Models (LLMs) can be used to perform **function calling**.

The goal is to take a natural-language request, determine which function from a predefined list should be called, and generate the corresponding parameters in a structured JSON format.

For example, given:

```text
"What's the weather in Paris?"
```

and a set of available functions:

```json
[
    {
        "name": "get_weather",
        "description": "Get the current weather of a city",
        "parameters": {
            "city": "string"
        },
        "returns": "string"
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "expression": "string"
        },
        "returns": "number"
    }
]
```

the system should identify:

```text
get_weather
```

and generate parameters such as:

```json
{
    "city": "Paris"
}
```

### Overview

The project uses a small local language model through a custom `llm_sdk`.

The generation process is divided into two main stages:

1. **Function name generation**

   * The LLM receives the user prompt and the available function definitions.
   * The generation is constrained so that the model selects one of the available function names.

2. **Parameter generation**

   * After selecting the function, the program retrieves its parameter definitions.
   * A finite-state machine controls the structure of the generated JSON.
   * Different parameter types use different generation strategies.

The project therefore explores how an LLM can be combined with deterministic programming techniques to produce structured outputs.

---

# Features

* Natural-language function selection.
* Constrained token generation.
* Logit masking using `-inf`.
* Greedy decoding using `argmax`.
* Finite-state machine for JSON generation.
* Support for:

  * `string`
  * `integer`
  * `number`
  * `boolean`
* Input validation using Pydantic.
* JSON output generation.
* Local LLM inference through a custom SDK.
* No external function-calling API is required.

---

# Algorithm Explanation

## 1. Constrained Decoding

The main algorithmic concept of this project is **constrained decoding**.

Normally, an autoregressive language model predicts a probability distribution over its entire vocabulary.

For example, suppose the vocabulary contains:

```text
["hello", "weather", "calculate", "Paris", ...]
```

The model produces a logit for every token:

```text
token       logit
-----------------
hello        1.2
weather      8.4
calculate    5.1
Paris        2.3
...
```

The token with the highest logit would normally be selected.

This project modifies this process by **masking tokens that are not allowed**.

For a disallowed token:

```python
logit = -np.inf
```

For an allowed token, the original logit is kept.

Conceptually:

```text
Original logits:

A → 4.2
B → 7.1
C → 2.4
D → 6.5

Allowed tokens:

A
C

After masking:

A → 4.2
B → -inf
C → 2.4
D → -inf
```

The model can therefore only select:

```text
A or C
```

The highest allowed token is then selected using:

```python
np.argmax(masked_logits)
```

This process is repeated token by token.

---

## 2. Function Name Generation

The first generation stage is implemented by `FunctionName`.

The model receives a prompt containing:

* the user's request;
* the available function names;
* their descriptions.

The model is asked to return only the name of the function that best matches the request.

Instead of allowing the model to generate arbitrary text, the program restricts the possible tokens to tokens belonging to the available function names.

### Example

Suppose the available functions are:

```text
get_weather
calculate
send_email
```

At a particular generation position, the program determines which token IDs can be used and masks all other vocabulary tokens.

The process is repeated until a complete function name is generated.

### Prefix filtering

An important improvement is to ensure that the generated prefix is still compatible with the available function names.

For example:

```text
Available:

get_weather
get_temperature
send_email
```

If the model generates:

```text
get_
```

then only functions beginning with:

```text
get_
```

should remain candidates.

After generating:

```text
get_w
```

only:

```text
get_weather
```

may remain.

This prevents the generator from combining tokens belonging to different function names.

A trie or equivalent prefix structure would be a suitable way to implement this more robustly.

---

# 3. Parameter Generation

After the function has been selected, `ParametersGenerator` generates its parameters.

The parameter generator uses a finite-state machine:

```text
START
  ↓
KEY
  ↓
COLON
  ↓
VALUE
  ↓
COMA
  ↓
KEY
  ↓
...
  ↓
END
  ↓
FINISH
```

The FSM determines what type of output is valid at each stage.

For example:

```json
{
    "city": "Paris",
    "temperature_unit": "C"
}
```

The program does not let the LLM freely generate the entire JSON structure.

Instead, structural elements such as:

```text
{
:
,
}
```

are generated deterministically.

Parameter names are also obtained from the function definition rather than asking the model to invent them.

The LLM is mainly used for generating parameter values.

---

# 4. Parameter Type Constraints

Each supported parameter type has its own generator.

## String

Strings are generated using `String`.

The generator detects the beginning and end of the string and handles quotation marks.

Because tokenization does not necessarily correspond to individual characters, string termination must be handled using the decoded generated text.

In particular, an escaped quote must not terminate the string.

For example:

```text
"hello \"world\""
```

The quotes around `world` are escaped and therefore do not terminate the JSON string.

An important implementation detail is that the escape character can be located in a **previous token** while the quote appears in the current token. Therefore, string processing must consider the accumulated decoded text rather than only the current token.

---

## Integer

`Integer` generates integer values using constrained token selection.

The first stage allows:

```text
-
0 1 2 3 4 5 6 7 8 9
```

The following generation stages continue to allow numerical characters.

Generation stops when a structural character such as:

```text
,
}
```

is encountered.

---

## Number

`Number` follows a similar approach but additionally allows a decimal point:

```text
-
0 1 2 3 4 5 6 7 8 9
.
```

The generator prevents multiple decimal points.

For example:

```text
42.5
```

is valid, while:

```text
42.5.7
```

is not.

If necessary, the implementation normalizes values such as:

```text
42
```

to:

```text
42.0
```

---

## Boolean

`Boolean` supports boolean-like values such as:

```text
true
false
```

and numeric representations:

```text
0
1
```

The generator constrains the model to the token IDs obtained from these possible representations.

---

# Design Decisions

## Why constrained decoding?

A normal LLM is designed to generate natural language.

Function calling requires a much more restricted output space.

For example, if the expected output is:

```json
{
    "city": "Paris"
}
```

allowing the model to freely generate text can result in:

```text
Sure! Here is the information you requested...
```

or malformed JSON.

Constrained decoding reduces the number of possible outputs and gives the programmer more control over the generation process.

---

## Why logit masking?

Logit masking is simple and directly compatible with autoregressive generation.

Instead of modifying the model itself, the program modifies its output distribution before selecting the next token.

This means the same language model can be reused while the application controls what it is allowed to generate.

---

## Why an FSM?

JSON generation has a predictable structure.

For example, after:

```text
{
    "city"
```

the next valid structural element is:

```text
:
```

After the colon, a value must be generated.

An FSM provides a simple way to represent these states explicitly:

```text
START
KEY
COLON
VALUE
COMA
END
FINISH
```

This makes the generation process easier to reason about than relying entirely on the LLM.

---

## Why separate generators for each type?

Different data types have different constraints.

For example:

```text
string  → arbitrary text
integer → digits
number  → digits + decimal point
boolean → true / false / 0 / 1
```

Separating these generators makes the implementation easier to understand and extend.

---

# Performance Analysis

## Accuracy

The system benefits from deterministic constraints because many invalid outputs can be prevented before they are generated.

However, the current implementation should not be considered a formal guarantee of perfectly valid output in every tokenizer/model situation.

In particular:

* tokenizer tokens can contain multiple characters;
* function-name selection must maintain prefix consistency;
* string escaping requires consideration of previous generated tokens;
* the model can still produce unexpected semantic values.

The final parameter string is passed through `json.loads()` to detect invalid JSON.

---

## Speed

The project uses greedy decoding:

```python
np.argmax(masked_logits)
```

This is computationally simpler than approaches such as beam search.

However, generation still requires repeated model inference for each generated token.

The total generation time therefore depends on:

* model size;
* number of input tokens;
* number of generated tokens;
* number of prompts;
* hardware;
* tokenizer performance.

The program measures the total execution time using:

```python
time.perf_counter()
```

and displays the elapsed time after processing all prompts.

---

## Reliability

Reliability is improved by combining:

```text
LLM
 ↓
Constrained decoding
 ↓
FSM
 ↓
Type-specific generators
 ↓
JSON validation
```

Each layer reduces the amount of freedom given to the model.

Nevertheless, reliability depends on the quality of the constraints and tokenizer behavior. More robust prefix-based function selection and grammar-aware token constraints would further improve the system.

---

# Challenges Faced

## Understanding LLM generation

One of the main challenges was understanding that an LLM does not directly generate complete words or characters.

It generates **tokens**.

The model first produces logits over its vocabulary, and the next token is selected from this distribution.

This was important for implementing token-level constraints.

---

## Tokenization

A major difficulty is that one character does not necessarily correspond to one token.

For example, a tokenizer may represent:

```text
"hello"
```

as one token or several tokens.

Therefore, constraints based on individual characters cannot automatically be assumed to work correctly at the token level.

This is particularly important for strings, numbers, and function names.

---

## Function-name consistency

Another challenge is making sure that tokens selected at different positions belong to the **same function name**.

For example, if:

```text
function_1 = get_weather
function_2 = send_email
```

the algorithm must not generate a sequence that combines a prefix from one function with a token from another.

This led to the need for prefix-based candidate elimination.

---

## Escaped characters

Handling strings is more complicated because an escaped quote does not terminate a JSON string:

```text
"hello \"world\""
```

The escape character may be generated in one token while the quote is generated in another.

Therefore, the implementation needs to consider the accumulated decoded text rather than only the current token.

---

## Combining deterministic logic with an LLM

Another challenge was deciding which parts should be controlled by the program and which parts should be generated by the model.

The project uses deterministic logic for:

* JSON structure;
* parameter names;
* type constraints;
* function candidates.

The LLM is mainly responsible for selecting the appropriate function and generating parameter values.

---

# Testing Strategy

Testing is performed at several levels.

## Input validation

The project uses Pydantic models to validate:

* prompt files;
* function definitions;
* required fields;
* supported parameter types.

Invalid input should be rejected before generation begins.

---

## Function selection

Function selection should be tested with prompts that clearly correspond to different functions.

For example:

```text
"What is the weather in Casablanca?"
```

should select a weather-related function.

Another prompt such as:

```text
"Send an email to Alice"
```

should select an email-related function.

Tests should also include functions with similar names to verify prefix filtering.

For example:

```text
get_weather
get_weather_forecast
get_temperature
```

---

## Parameter generation

Parameter generators should be tested independently with:

* positive integers;
* negative integers;
* decimal numbers;
* strings;
* escaped quotes;
* boolean values;
* multiple parameters.

---

## JSON validation

Generated parameter strings are passed to:

```python
json.loads()
```

to detect malformed JSON.

This ensures that invalid JSON is detected before it is written to the output file.

---

## End-to-end testing

The complete pipeline can be tested using:

```text
Input JSON
    ↓
Parser
    ↓
Function selection
    ↓
Parameter generation
    ↓
JSON validation
    ↓
Output JSON
```

This verifies that all components work together correctly.

---

# Example Usage

## Input

A function definition file can contain functions such as:

```json
[
    {
        "name": "get_weather",
        "description": "Get the weather of a city",
        "parameters": {
            "city": "string"
        },
        "returns": "string"
    }
]
```

A prompt file can contain:

```json
[
    {
        "prompt": "What is the weather in Paris?"
    }
]
```

## Expected result

The system should identify:

```text
get_weather
```

and generate parameters similar to:

```json
{
    "city": "Paris"
}
```

The final output is written to the configured output file.

---

# Installation

Clone the project and enter the project directory:

```bash
git clone <repository-url>
cd call-me-maybe
```

Create and activate a virtual environment if required by the project setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies using the project's package manager.

For example, with `uv`:

```bash
uv sync
```

The project also requires the local `llm_sdk` package used to load the language model.

---

# Execution

The program can be executed using the project's configured entry point.

If running the package directly:

```bash
python3 -m src
```

The default input files are:

```text
data/input/function_calling_tests.json
data/input/functions_definition.json
```

The default output file is:

```text
data/output/function_calls.json
```

The input and output paths can be changed through command-line arguments.

Example:

```bash
python3 -m src \
    --input data/input/function_calling_tests.json \
    --functions_definition data/input/functions_definition.json \
    --output data/output/function_calls.json
```

---

# Project Structure

A simplified project structure is:

```text
.
├── data
│   ├── input
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output
│       └── function_calls.json
│
├── src
│   ├── engine.py
│   ├── function_name.py
│   ├── function_parameters.py
│   ├── gen_boolean.py
│   ├── gen_integers.py
│   ├── gen_number.py
│   ├── gen_strings.py
│   ├── validation_classes.py
│   ├── validation_input.py
│   └── main.py
│
├── llm_sdk
│   └── ...
│
├── pyproject.toml
└── README.md
```

---

# Resources

The following resources were useful for understanding the concepts behind the project:

* **Attention Is All You Need** — original Transformer architecture paper.
* **Hugging Face Transformers documentation** — understanding model inference and transformer models.
* **Hugging Face Tokenizers documentation** — understanding tokenization and token IDs.
* **NumPy documentation** — numerical operations and `argmax`.
* **Python JSON documentation** — JSON encoding and decoding.
* **Pydantic documentation** — data validation and structured input validation.
* **Python Enum documentation** — implementation of finite-state machines.
* **Finite State Machines** — general concepts for representing state-based algorithms.
* **Constrained decoding literature** — techniques for restricting the output space of language models.

---

# AI Usage

AI tools were used as a learning and development aid during this project.

They were used to:

* understand fundamental concepts related to AI, machine learning, deep learning, LLMs and tokenization;
* understand how transformer-based language models generate tokens;
* understand logits, vocabulary IDs and greedy decoding;
* study constrained decoding and logit masking;
* reason about finite-state machines;
* investigate Python implementation and typing issues;
* discuss possible approaches to improve the generation algorithms;
* review parts of the project documentation and README.

The implementation, testing and debugging of the project were performed by the project author.

AI-generated suggestions were treated as explanations and development assistance rather than as a replacement for understanding the implementation.

---

# Limitations and Possible Improvements

The current implementation can be improved in several areas.

### Function-name generation

A prefix tree (trie) could be used to guarantee that every generated token remains compatible with at least one complete function name.

### String generation

String generation could maintain a complete decoded character buffer and correctly handle escaped characters across token boundaries.

### Grammar-based decoding

The current FSM could be extended into a more formal grammar-based constrained decoder.

### Token-aware constraints

The constraints could be designed around the tokenizer's actual vocabulary rather than assuming that encoding individual characters always corresponds to individual character tokens.

### Validation

The final generated parameters could also be validated against the selected function's parameter schema using Pydantic or another schema-validation mechanism.

### Decoding strategies

The project currently uses greedy decoding. Other strategies such as beam search could be investigated and compared.

---

# Conclusion

**Call Me Maybe** demonstrates how an LLM can be combined with classical deterministic algorithms to perform structured function calling.

The central idea is to reduce the freedom of the language model by controlling its output during decoding.

The project combines:

```text
LLM
 │
 ├── Tokenization
 │
 ├── Logits
 │
 ├── Constrained decoding
 │      └── Logit masking
 │
 ├── Function selection
 │
 └── Parameter generation
        └── Finite-state machine
              ├── String
              ├── Integer
              ├── Number
              └── Boolean
```

The project provided practical experience with LLM inference, tokenization, constrained generation, finite-state machines, structured data generation and the interaction between probabilistic models and deterministic software.
