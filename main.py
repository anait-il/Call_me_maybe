from pathlib import Path
from validation_input import validate_prompt, validate_def
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from typing import Dict, Any, List
import numpy as np


def build_prompt(user_prompt: str, function: List[str]) -> str:
        return f"""
            You are a function-calling assistant that
            helps me get a JSON format from a user prompt.

            Available functions:
            {function}

            Example:

            Prompt: "what is the sum of 1 and 2"

            Answer:
            {{
                "prompt": "what is the sum of 1 and 2",
                "name": "fn_add_numbers",
                "parameters": {{"a": 1.0, "b": 2.0}}
            }}

            User prompt: {user_prompt}
        """

def main():
    print("Hello from call-me-maybe!")
    try:
        prompts: Dict[str, str] = validate_prompt("data/input/function_calling_tests.json")
        functions: Dict[str, Any] = validate_def("data/input/functions_definition.json")

    except (ValueError, ValidationError):
        pass

    model = Small_LLM_Model()
    fc = [func['name'] for func in functions]
    output = []
    for prompt in prompts:
        tokens: List[int] = np.array(model.encode(build_prompt(prompt['prompt'], fc))).tolist()
        while len(output) < len(max(fc)):
            logit = model.get_logits_from_input_ids(tokens[0]) # list of the tokens with scores
            next_id = np.argmax(logit) # id of next char
            output += [next_id]
            tokens[0] += [next_id]

        print(model.decode(output))


if __name__ == "__main__":
    main() 
