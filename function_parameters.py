from llm_sdk import Small_LLM_Model  # type: ignore
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

        print(f"function param {self.generated_output}")
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

        return self.__get_allowed_ids()

    def __add_couts(self, current_key: str) -> str:

        text: str = ""
        text += "\""
        text += current_key
        text += "\""
        return text

    def __get_allowed_ids(self) -> str:

        if self.current_state == State.START:
            return "{"

        if self.current_state == State.COLON:
            return ":"

        if self.current_state == State.END:
            return "}"

        if self.current_state == State.COMA:
            return ","

        return ""

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

    def __get_parameters_definition(self,
                                    parameteres: Dict[str, Dict[str, str]]) -> Dict[str, str]:

        param: Dict[str, str] = {}
        for key, value in parameteres.items():
            param[key] = value["type"]

        return param

    def __build_prompt(self,
                       user_prompt: str,
                       parameters: Dict[str, Dict[str, str]]) -> str:

        function_definition: Dict[str, Dict[str, str]] = (
            {self.function_name: {
                "parameters": self.__get_parameters_definition(parameters)
            }}
        )

        return f"""
            You are a function-calling assistant that
            helps me get a JSON format from a user prompt.

            Available functions:
            {function_definition}

            Example:

            Prompt: "what is the sum of 1 and 2"

            Answer:
            {{
                "prompt": "what is the sum of 1 and 2",
                "name": "fn_add_numbers",
                "parameters": {{"a": 1.0, "b": 2.0}}
            }}

            User prompt: {user_prompt}

            JSON:
            {{
                "prompt": {user_prompt},
                "name": "'''{self.function_name}
                "parameters": 
        """
