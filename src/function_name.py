import numpy as np
from numpy.typing import NDArray
from itertools import count
from typing import Dict, List, Any
from llm_sdk import Small_LLM_Model  # type: ignore


class FunctionName:
    """Select a function name using constrained LLM generation."""

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: str,
                 functions_definition: List[Dict[str, Any]]) -> None:
        """Initialize the function name generator.

        Args:
            model: Language model used for function selection.
            prompt: User request used to select a function.
            functions_definition: Available function definitions.
        """

        self.model: Small_LLM_Model = model
        self.prompt: str = prompt
        self.functions_definition: List[Dict[str, Any]] = functions_definition
        self.available_functions: List[str] = (
            [func['name'] for func in self.functions_definition])
        self.functions_token: List[List[int]] = (
            [np.array(self.model.encode(x))[0].tolist()
             for x in self.available_functions])

    def set_allowed_ids(self, index: int, gen_tokens: List[int]) -> List[int]:
        """Get token IDs that can continue the current function name.

        Args:
            index: Current token position.
            gen_tokens: Tokens generated so far.

        Returns:
            Token IDs allowed by the function-name constraints.
        """

        ids: List[int] = []

        for tokens in self.functions_token:

            prefix_length: int = len(gen_tokens)

            if tokens[:prefix_length] == gen_tokens:
                if len(tokens) > index:
                    ids.append(tokens[index])

        if gen_tokens in self.functions_token:
            ids.extend(list(self.model.encode("\"")[0]))

        return ids

    def generate_function_name(self) -> str:
        """Generate a function name using constrained decoding.

        Returns:
            The selected function name.
        """

        output: str = ""
        tokens: List[int] = np.array(self.model.encode(
            self.build_prompt(self.prompt)))[0].tolist()
        tokens += list(self.model.encode("\"")[0])
        generated_tokens: List[int] = []

        for i in count():
            logits: List[float] = self.model.get_logits_from_input_ids(tokens)
            mask: NDArray = np.full_like(logits, float("-inf"))
            allowed_ids: List[int] = self.set_allowed_ids(i, generated_tokens)
            mask[allowed_ids] = 0
            masked_logits: NDArray = mask + logits
            next_id: int = int(np.argmax(masked_logits))

            if self.model.decode([next_id]) == "\"":
                break

            output += self.model.decode([next_id])

            tokens.append(next_id)
            generated_tokens.append(next_id)

        return output

    def build_prompt(self, user_prompt: str) -> str:
        """Build the function-selection prompt.

        Args:
            user_prompt: User request to include in the prompt.

        Returns:
            A formatted prompt for function selection.
        """

        function_description = []
        for function in self.functions_definition:

            function_description.append(
                [function['name'], function['description']])

        return f"""
You are a function selector.

Your task:
- Read the user request.
- Choose the BEST function by reading the description.
- Return ONLY the function name.
- Do not explain anything.
- If no function matches, return: NONE

available functions:
{self.available_functions}

Description of the functions:
{function_description}

user request:
{user_prompt}

name: "'''
"""
