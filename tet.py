from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

prompt = "Reverse the string 'hello'"
functions_definition = {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }

function_name = "fn_reverse_string"







def build_prompt(user_prompt: str,
                   parameters) -> str:

        function_description: str = ""
        for function in functions_definition:

            if function['name'] == function_name:
                function_description = function['description']
                break
        striped_param = (
            {key: value["type"]
                for key, value in parameters.items()})
        return f"""
You extract function parameter values from a user's request.

Function name:
{function_name}

Function description:
{function_description}

Required parameters:
{striped_param}

Your task:
Read the user's request and extract the value of each required parameter.

Rules:
1. Return ONLY a JSON object.
2. The JSON keys MUST be the parameter names listed in Required parameters.
3. The values MUST match the required parameter types.
4. Do not add parameters that are not listed.
5. Do not add explanations, comments, or extra text.
6. Extract values exactly from the user's request when possible.
7. Do not invent values that are not present in the user's request.

Examples:

Example 1
Q: Reverse the string "red"
A: {{"s": "red"}}

Example 2
Q: What is the sum of 2 and 3?
A: {{"a": 2, "b": 3}}

Example 3
Q: Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS
A: {{"source_string": \
    "Hello 34 I'm 233 years old", \
        "regex": "\\d+", "replacement": "NUMBERS"}}

Example 4
Q: Replace all vowels in 'Programming is fun' with "$"
A: {{source_string: \
    "Programming is fun", \
        "regex": "[aeiouAEIOU]" \
            "replacement": "$"}}

User request:
Q: {user_prompt}

A:
"""

user_prompt = build_prompt(prompt, functions_definition["parameters"])
tokens = model.encode(user_prompt)

for _ in range(100):
    logits = model.get_logits_from_input_ids(tokens)
    max_score = max(logits)
    next_token = (int(x) for x, s in enumerate(logits) if s == max_score)
    print(model.decode(next_token))


