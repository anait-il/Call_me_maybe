import numpy as np
from typing import Dict, List, Any
from llm_sdk import Small_LLM_Model

class FunctionName:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompts: Dict[str, str],
                 functions_definition: Dict[str, Any])-> None:
        self.model: Small_LLM_Model = model
        self.prompts: Dict[str, str] = prompts
        self.functions_definition: Dict[str, Any] = functions_definition

    def get_function_name(self)-> str:

        available_functions = [func['name'] for func in self.functions_definition]

        for prompt in self.prompts:

            output = ''
            tokens: List[int] = np.array(self.model.encode(
                self.build_prompt(prompt['prompt'],
                                  available_functions))).tolist()
            allowed_ids: List[int] = np.array(self.model.encode(available_functions)).tolist()

            while True:

                logits: List[float] = self.model.get_logits_from_input_ids(tokens[0])
                mask: List[float] = np.full_like(logits, float('-inf'))
                mask[allowed_ids] = 0
                masked_logits = mask + logits
                next_id = np.argmax(masked_logits)
                output += self.model.decode([next_id])
                if len(output) >= len(max(available_functions)):
                    break
                tokens[0] += [next_id]                
                print("##" * 20, end='\n\n')
                print(output)

    def build_prompt(self, user_prompt: str, functions_list: List[str])-> str:
        return f"""
            You are a function-calling assistant that
            helps me get a function name from a user prompt.

            Available functions:
            {functions_list}

            Example:

            Prompt: "what is the sum of 1 and 2"

            Answer: "fn_add_number"

            User prompt: {user_prompt}
        """