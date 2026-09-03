from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np


class Fsm(Enum):
    DIGITS = 0
    END = 1


class Boolean:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: str)-> None:

        self.__model: Small_LLM_Model = model
        self.current_state: Fsm = Fsm.DIGITS
        self.prompt: List[int] = prompt

    def generate_bool(self)-> List[int]:

        self.generated: List[int] = []
        while not self.current_state == Fsm.END:

            logits: List[int] = self.__model.get_logits_from_input_ids(self.prompt)
            mask: List[int] = np.full_like(logits, float("-inf"))
            allowed_tokens: List[int] = self.__get_tokens(self.current_state)
            mask[allowed_tokens] = 0
            masked_logits: List[int] = mask + logits
            next_token: int = np.argmax(masked_logits)
            self.generated.append(next_token)
            self.prompt.append(next_token)
            if self.__my_decode(self.generated) == 'true' or \
                self.__my_decode(self.generated) == "false":
                break

        return self.generated

    def __get_tokens(self, state: Fsm)-> List[int]:

        tokens: List[int] = []

        if state == Fsm.DIGITS:
            tokens = np.array(self.__model.encode("truefalse")).tolist()[0]
            self.current_state = Fsm.END
            return tokens

    def __my_decode(self, tokens: List[int] | int)-> str:

        if not isinstance(tokens, List):
            tokens = list(tokens)
        return self.__model.decode(tokens)

    def __my_encode(self, string: str)-> List[int]:

        return np.array(self.__model.encode(string)).tolist()[0]
