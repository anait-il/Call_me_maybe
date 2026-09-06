from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np


class Fsm(Enum):
    DIGITS = 1
    ALPHANUM = 2
    END = 3


class Number:

    def __init__(self,
                 model: Small_LLM_Model,
                 building_prompt: List[int],
                 user_prompt: str)-> None:

        self.__model: Small_LLM_Model = model
        self.building_prompt: List[int] = building_prompt
        self.user_prompt: str = user_prompt

    def generate_numbers(self)-> str:

        self.__generated: str = ""
        self.__generated_tokens: List[int] = []
        self.__current_state: Fsm = Fsm.DIGITS
        while not self.__current_state == Fsm.END:

            if len(self.__generated) > len(self.user_prompt):
                break
            logits: List[int] = self.__model.get_logits_from_input_ids(
                (self.building_prompt + self.__generated_tokens))
            masked_logits: List[int] = self.__get_masked_logits(logits)
            next_token: int = np.argmax(masked_logits)
            if self.__current_state == Fsm.DIGITS:
                self.__current_state = Fsm.ALPHANUM

            elif next_token in self.__my_encode(",}"):
                self.__current_state = Fsm.END
                break

            self.__generated += self.__model.decode(next_token)
            self.__generated_tokens += [next_token]

        self.__convert_to_float()
        return self.__generated

    def __get_masked_logits(self, logits: List[int])-> List[int]:

        mask: List[int] = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.__current_state)
        mask[allowed_tokens] = 0

        return mask + logits

    def __get_tokens(self, state: Fsm)-> List[int]:

        tokens: List[int] = []

        if state == Fsm.DIGITS:
            tokens += np.array(self.__model.encode("-0123456789")).tolist()[0]
            return tokens

        elif state == Fsm.ALPHANUM:
            tokens = np.array(self.__model.encode("0123456789.,}")).tolist()[0]
            return tokens

    def __convert_to_float(self)-> None:
        if "." not in self.__generated:
            self.__generated += "."
            self.__generated += "0"

        if self.__generated[-1] == ".":
            self.__generated += "0"
     
    def __my_encode(self, string: str)-> List[int]:
        return np.array(self.__model.encode(string)).tolist()[0]
