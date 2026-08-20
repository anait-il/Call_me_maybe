import numpy as np
from itertools import count
from typing import Dict, List, Any
from llm_sdk import Small_LLM_Model

class FunctionName:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompts: Dict[str, str],
                 functions_definition: Dict[str, Any])-> None:
        self.model: Small_LLM_Model = model
        self.prompts: List[Dict[str, str]] = prompts
        self.functions_definition: List[Dict[str, Any]] = functions_definition
        self.available_functions: List[str] = [func['name']
                                               for func in self.functions_definition]
        self.functions_token: List[List[int]] = [np.array(self.model.encode(x))[0].tolist()
                                                 for x in self.available_functions] 

    def set_allowed_ids(self, index: int)-> List[int]:

        ids: List[int] = []

        for tokens in self.functions_token:
            if len(tokens) > index:

                ids.append(tokens[index])

        return ids

    def get_function_name(self, prompt: str)-> str: 

        output: str = ""
        
        tokens: List[int] = np.array(self.model.encode(
            self.build_prompt(prompt)))[0].tolist()

        for i, _ in enumerate(count()):
            logits: List[float] = self.model.get_logits_from_input_ids(tokens)
            mask: List[float] = np.full_like(logits, float("-inf"))
            allowed_ids: List[int] = self.set_allowed_ids(i)
            mask[allowed_ids] = 0
            masked_logits = mask + logits
            next_id = np.argmax(masked_logits)
            output += self.model.decode([next_id])
            if output in self.available_functions:
                break

            if len(output) >= len(max(self.available_functions)):
                break

            tokens.append(next_id)

        return output

    def generate_function_name(self)-> None:

        for prompt in self.prompts:

            output: str = self.get_function_name(prompt['prompt'])
            print(output)

    def build_prompt(self, user_prompt: str)-> str:
        fn = []
        for function in self.functions_definition:

            fn.append([function['name'], function['description']])

        return f"""
You are a function selector.

Your task:
- Read the user request.
- Choose the BEST function.
- Return ONLY the function name.
- Do not explain anything.
- If no function matches, return: NONE

avialable functions:
{fn}

user request:
{user_prompt}
"""
