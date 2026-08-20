from llm_sdk import Small_LLM_Model
from typing import Dict, List, Any
from enum import Enum
from itertools import count
import numpy as np

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
                 function_name: str ,functions_definition: List[Dict[str, Any]])-> None:
        self.model: Small_LLM_Model = model
        self.prompt: str = prompt
        self.function_name: str = function_name
        self.functions_definition: List[Dict[str, Any]] = function_definition
        self.current_state = State.START


    def transition(self, token_id: int)-> bool:
       ...

    def get_tokens_for(self, input: str)-> List[int]:

        return np.array(self.model.encode(input))[0].tolist()

    def get_type(self)-> str:

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
            for key in self.parameters.keys():
                param += '"'
                param += key
                param += '"'

            return self.get_tokens_for(param)

        def get_tokens_for_param_types(data_type: str)-> List[int]:

            if data_type == "integer":
                return self.get_tokens_for("0123456789")

            if data_type == "number":
                return self.get_tokens_for("0123456789.")

            if data_type == "boolean":
                return self.get_tokens_for("TrueFalse")

            if data_type == "string":
                return self.get_tokens_for("")

        if self.current_state == State.VALUE:
            return value_tokens(self.get_type())

    def get_function_definition(self)-> None:

        for function in self.function_definition:

            if function['name'] == self.function_name:
                return function

        raise ValueError(f"Unknown function: {self.function_name}")    

    def generate_parameter(self)-> None:

        self.generated: str = ""
        function: Dict[str, Any] = self.get_function_definition()
        self.parameters: Dict[str, Dict[str, str]] = function['parameters']

        number_of_params = len(paramters)
        tokens: List[int] = np.array(self.model.encode(self.build_prompt(self.prompt)))[0].tolist()
        
        while self.current_state != State.FINISH:

            self.generated += self.get_next_element(self.current_state, tokens)

    def get_next_element(self, state: State, prompt_ids: List[int])-> str:

        element: List[int] = []

        while self.current_state == state:

            logits = self.model.get_logits_from_input_ids(prompt_ids)
            mask = np.full_like(logits, float("-inf"))
            allowed_ids = self.get_allowed_ids()
            if not allowed_ids: 
                self.get_value_ids(self.get_type())

            mask[allowed_ids] = 0
            masked_logits = maks + logits
            next_token = np.argmax(masked_logits)
            prompt_ids.append(next_token)
            element.append(next_token)
            self.transition(next_token)
        
        return self.model.decode([element])
