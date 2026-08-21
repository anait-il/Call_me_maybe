from llm_sdk import Small_LLM_Model
from typing import Dict, List, Any
from enum import Enum
import numpy as np
import json


class State(Enum):
    START = 0
    KEY = 1
    COLON = 2
    VALUE = 3
    COMA = 4
    END = 5
    FINISH = 6


class Parameters:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: str,
                 function_name: str ,
                 functions_definition: List[Dict[str, Any]])-> None:
        self.model: Small_LLM_Model = model
        self.prompt: str = prompt
        self.function_name: str = function_name
        self.functions_definition: List[Dict[str, Any]] = functions_definition
        self.current_state = State.START


    def next_state(self, output: str)-> State:

        if self.current_state == State.START:
            return State.KEY
        if self.current_state == State.KEY:
            return State.COLON
        if self.current_state == State.COLON:
            return State.VALUE
        if self.current_state == State.VALUE:
            if len(output.split(",")) == self.number_of_param:
                return State.END
            else:
                return State.COMA
        if self.current_state == State.COMA:
            return State.KEY
        if self.current_state == State.END:
            return State.FINISH

    def __transition(self, token_id: int, output: str)-> None:

        if self.current_state == State.VALUE:
           if token_id == self.model.encode('"'):
                if output[-2] != "\\":
                    self.current_state = self.next_state(output)
                    return None

        if self.current_state == State.KEY:
            output = self.model.decode(output)
            if output[-1] == "\"":
                if output[-2] in list(self.parameters.keys()):
                    self.current_state = self.next_state(output)
                    return None

        if self.current_state == State.COLON:
            self.current_state = self.next_state(output)
            return None

        if self.current_state == State.COMA:
            self.current_state = self.next_state(output)
            return None

        if self.current_state == State.START:
            self.current_state = self.next_state(output)
            return None

        if self.current_state == State.END:
            self.current_state = self.next_state(output)
            return None

    def __get_tokens_for(self, input: str)-> List[int]:

        return np.array(self.model.encode(input))[0].tolist()

    def get_tokens_for_bool(self)-> List[int]:

        tokens: List[int] = []
        tokens.append(np.array(self.model.encode("true")[0].tolist()))
        tokens.append(np.array(self.model.encode("false")[0].tolist()))
        return tokens

    def __get_type(self)-> str:

        lst_var = self.generated.split(",")[-1]
        var = lst_var.split[':'][0]
        return self.paramters[var.strip()]['type']

    def get_allowed_ids(self)-> List[int]:

        if self.current_state == State.START:
            return self.get_tokens_for("{")

        if self.current_state == State.COLON:
            return self.get_tokens_for(":")

        if self.current_state == State.END:
            return self.get_tokens_for("}")
        
        if self.current_state == State.COMA:
            return self.get_tokens_for(",")

        if self.current_state == State.KEY:
            param: str = ""
            for key in list(self.parameters.keys()):
                param += '\"'
                param += key

            return self.get_tokens_for(param)

        if self.current_state == State.VALUE:
            return self.__value_type

    def __get_function_definition(self)-> None:

        for function in self.functions_definition:

            if function['name'] == self.function_name:
                return function

        raise ValueError(f"Unknown function: {self.function_name}")    

    def generate_parameter(self)-> None:

        self.generated: str = ""
        function: Dict[str, Any] = self.get_function_definition()
        self.parameters: Dict[str, Dict[str, str]] = function['parameters']

        self.number_of_param: int = len(self.parameters)
        tokens: List[int] = np.array(self.model.encode(self.build_prompt(self.prompt)))[0].tolist()
        
        while self.current_state != State.FINISH:
            output = self.get_next_element(self.current_state, tokens)
            self.generated += output
            print(self.generated)

    def get_next_element(self, state: State, prompt_ids: List[int])-> str:

        element: List[int] = []

        while self.current_state == state:

            logits = self.model.get_logits_from_input_ids(prompt_ids)
            mask = np.full_like(logits, float("-inf"))
            allowed_ids = self.get_allowed_ids()
            if allowed_ids:
                mask[allowed_ids] = 0
                logits = mask + logits
            next_token = np.argmax(logits)
            prompt_ids.append(next_token)
            element.append(next_token)
            self.transition(next_token, element)

        return self.model.decode(element)

    def build_prompt(self, user_prompt: str)-> str:

        function_description: str = ""
        for function in self.functions_definition:

            if function['name'] == self.function_name:
                function_description = function['description']
                break

        return f"""
You are a parameter extraction system.

Your task is to extract the parameter values required by the selected function from the user's request.

Selected function:
{self.function_name}

Function description:
{function_description}

Required parameters:
{self.parameters}

Rules:
- Extract only the parameters required by the selected function.
- Use exactly the parameter names provided in the definition.
- The value of each parameter must match its required type.
- Do not add extra parameters.
- Do not calculate or execute the function.
- Do not return the function name.
- Do not explain your answer.
- Return only a JSON object containing the parameters.
- The JSON object must contain all required parameters.

User request:
{user_prompt}

Output:
"""
