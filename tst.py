from llm_sdk import Small_LLM_Model
import numpy as np

def main():

    model = Small_LLM_Model()
    print(model.encode("TrueFalse").tolist()[0])
    print(model.decode([2514]))
    exit(1)

    user_prompt = "what is the sum of 2 and 3"
    parameters = {"a": "number", "b": "number"}
    function_description = "Add two numbers together and return their sum."
    function_name = "fn_add_number"

    output = ""
    prompt = f"""
You extract function parameter values from a user's request.

Function name:
{function_name}

Function description:
{function_description}

Required parameters:
{parameters}

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
Q: What is the sum of 2 and 3
A: {{"a": 2, "b": 3}}

Example 2
Q: Reverse the string 'hello'
A: {{"s": "hello"}}

Example 3
Q: Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS
A: {{"source_string": "Hello 34 I'm 233 years old", "regex": "34 233", "replacement": "NUMBERS"}}

User request:
Q: {user_prompt}

A: 
"""
    
    while not output or output.strip()[-1] != "}":
        tokens = model.encode(prompt).tolist()[0]
        logits = model.get_logits_from_input_ids(tokens)
        id = np.argmax(logits)
        output += model.decode(id)
        prompt += output
        print(output)

    print("#" * 99)
    print(output)
    print("#" * 99)


if __name__ == "__main__":
     main()