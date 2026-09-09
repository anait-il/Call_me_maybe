from llm_sdk import Small_LLM_Model
from typing import Dict, List, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from gen_integers import Integer
from gen_boolean import Boolean
from gen_number import Number
from gen_strings import String


class State(Enum):
    START = 0
    KEY = 1
    COLON = 2
    VALUE = 3
    COMA = 4
    END = 5
    FINISH = 6


class ParametersGenerator:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: str,
                 function_name: str,
                 functions_definition: List[Dict[str, Any]]) -> None:

        self.__model: Small_LLM_Model = model
        self.prompt: str = prompt
        self.function_name: str = function_name
        self.functions_definition: List[Dict[str, Any]] = functions_definition

    def __get_function_defintion_parameters(self) -> Dict[str, Dict[str, str]]:

        for function in self.functions_definition:

            if function['name'] == self.function_name:
                return dict(function['parameters'].copy())

        raise ValueError("[Error] Unkown function name: "
                         f"must be one of '{self.functions_definition}'")

    def generate_parameter(self) -> str:

        self.generated_output: str = ""
        self.current_state: State = State.START
        parameters: Dict[str, Dict[str, str]] = (
            self.__get_function_defintion_parameters())
        prompt_tokens: List[int] = np.array(
            self.__model.encode(
                self.__build_prompt(self.prompt, parameters)))[0].tolist()

        while self.current_state != State.FINISH:

            output = self.__get_next_token(self.current_state,
                                           prompt_tokens,
                                           parameters)
            if not output:
                self.current_state = State.END
                continue

            elif self.current_state == State.END:
                self.current_state = State.FINISH

            elif self.current_state == State.START:
                self.current_state = State.KEY

            elif self.current_state == State.KEY:
                self.current_state = State.COLON

            elif self.current_state == State.COLON:
                self.current_state = State.VALUE

            elif self.current_state == State.VALUE:
                if not parameters:
                    self.current_state = State.END
                else:
                    self.current_state = State.COMA

            elif self.current_state == State.COMA:
                self.current_state = State.KEY

            self.generated_output += output
            prompt_tokens += np.array(self.__model.encode(output)).tolist()[0]

        return self.generated_output

    def __get_next_token(self,
                         state: State,
                         prompt_tokens: List[int],
                         parameters: Dict[str, Dict[str, str]]) -> str | None:

        output: List[int] = []
        if state == State.KEY:
            if not parameters:
                return None

            self.current_key: str = list(parameters.keys())[0]
            self.current_value: Dict[str, str] = parameters[self.current_key]
            del parameters[self.current_key]
            return self.__add_couts(self.current_key)

        if state == State.VALUE:
            return self.__get_parameter_value(prompt_tokens)

        logits: List[float] = (
            self.__model.get_logits_from_input_ids(prompt_tokens))
        masked_logits: NDArray = self.__get_masked_logits(logits)
        next_token: int = int(np.argmax(masked_logits))

        output.append(next_token)

        return self.__model.decode(output)

    def __get_masked_logits(self,
                            logits: List[float]) -> NDArray:

        mask: NDArray = np.full_like(logits, float("-inf"))
        allowed_ids: List[int] = self.__get_allowed_ids()
        mask[allowed_ids] = 0

        return mask + logits

    def __add_couts(self, current_key: str) -> str:

        text: str = ""
        text += "\""
        text += current_key
        text += "\""
        return text

    def __get_allowed_ids(self) -> List[int]:

        if self.current_state == State.START:
            return self.__get_tokens_for("{")

        if self.current_state == State.COLON:
            return self.__get_tokens_for(":")

        if self.current_state == State.END:
            return self.__get_tokens_for("}")

        if self.current_state == State.COMA:
            return self.__get_tokens_for(",")

        return []

    def __get_parameter_value(self, prompt_tokens: List[int]) -> str:

        if self.current_value['type'] == "string":
            string = String(self.__model, prompt_tokens, self.prompt)
            return string.generate_string()

        if self.current_value['type'] == "number":
            number = Number(self.__model, prompt_tokens, self.prompt)
            return number.generate_numbers()

        if self.current_value['type'] == "integer":
            integer = Integer(self.__model, prompt_tokens, self.prompt)
            return integer.generate_integer()

        if self.current_value['type'] == "boolean":
            boolean = Boolean(self.__model, prompt_tokens, self.prompt)
            return boolean.generate_bool()

        return ""

    def __get_tokens_for(self, input: str) -> List[int]:
        return [int(x)
                for x in np.array(self.__model.encode(input))[0]]

    def __build_prompt(self,
                       user_prompt: str,
                       parameters: Dict[str, Dict[str, str]]) -> str:

        function_description: str = ""
        for function in self.functions_definition:

            if function['name'] == self.function_name:
                function_description = function['description']
                break
        striped_param: Dict[str, str] = (
            {key: value["type"]
                for key, value in parameters.items()})
        return f"""
You extract function parameter values from a user's request.

Function name:
{self.function_name}

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
