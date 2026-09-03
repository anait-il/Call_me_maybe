from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np


class Fsm(Enum):
    SIGN = 0
    DIGITS = 1
    ALPHANUM = 2
    END = 3


class Integer:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: str)-> None:

        self.__model: Small_LLM_Model = model
        self.current_state: Fsm = Fsm.SIGN
        self.prompt: List[int] = prompt

    def generate_integer(self)-> List[int]:

        self.generated: List[int] = []
        while not self.current_state == Fsm.END:

            logits: List[int] = self.__model.get_logits_from_input_ids(self.prompt)
            mask: List[int] = np.full_like(logits, float("-inf"))
            allowed_tokens: List[int] = self.__get_tokens(self.current_state)
            mask[allowed_tokens] = 0
            masked_logits: List[int] = mask + logits
            next_token: int = np.argmax(masked_logits)
            if next_token in self.__my_encode(",}"):
                self.current_state = Fsm.END
                break

            self.generated.append(next_token)
            self.prompt.append(next_token)
            if self.__my_encode("+") in self.generated:
                self.generated.remove("+")

        return self.generated

    def __get_tokens(self, state: Fsm)-> List[int]:

        tokens: List[int] = []

        if state == Fsm.SIGN:
            tokens = np.array(self.__model.encode("-+")).tolist()[0]
            self.current_state = Fsm.DIGITS
            return tokens

        if state == Fsm.DIGITS:
            tokens = np.array(self.__model.encode("0123456789")).tolist()[0]
            self.current_state = Fsm.ALPHANUM
            return tokens

        if state == Fsm.ALPHANUM:
            tokens = np.array(self.__model.encode("0123456789,}")).tolist()[0]
            return tokens


    def __my_decode(self, tokens: List[int] | int)-> str:

        if not isinstance(tokens, List):
            tokens = list(tokens)
        return self.__model.decode(tokens)

    def __my_encode(self, string: str)-> List[int]:
    
        return np.array(self.__model.encode(string)).tolist()[0]

        