from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np


class Fsm(Enum):
    SIGN = 0
    DIGITS = 1
    DOT = 2
    ALPHANUM = 3
    END = 4


class Number:

    def __init__(self,
                 model: Small_LLM_Model,
                 building_prompt: List[int],
                 user_prompt: str)-> None:

        self.__model: Small_LLM_Model = model
        self.building_prompt: List[int] = building_prompt
        self.user_prompt: str = user_prompt

    def generate_numbers(self)-> str:

        self.generated: str = ""
        self.current_state: Fsm = Fsm.SIGN
        while not self.current_state == Fsm.END:

            if len(self.generated) > len(self.user_prompt):
                break

            logits: List[int] = self.__model.get_logits_from_input_ids(self.building_prompt)
            masked_logits: List[int] = self.__get_masked_logits(logits)
            next_token: int = np.argmax(masked_logits)

            if self.current_state == Fsm.SIGN:
                self.current_state = Fsm.DIGITS
                if self.__model.decode([next_token]) == "+":
                    continue

            elif self.current_state == Fsm.DIGITS:
                self.current_state = Fsm.DOT

            elif self.__model.decode([next_token]) == ".":
                self.current_state = Fsm.ALPHANUM

            elif next_token in self.__my_encode(",}"):
                self.current_state = Fsm.END
                break

            self.generated += self.__model.decode(next_token)
            self.building_prompt.append(next_token)

        self.__convert_to_float()
        return self.generated

    def __get_masked_logits(self, logits: List[int])-> List[int]:

        mask: List[int] = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.current_state)
        mask[allowed_tokens] = 0

        return logits + mask

    def __get_tokens(self, state: Fsm)-> List[int]:

        tokens: List[int] = []

        if state == Fsm.SIGN:
            tokens = np.array(self.__model.encode("-+")).tolist()[0]
            return tokens

        if state == Fsm.DIGITS:
            tokens = np.array(self.__model.encode("0123456789")).tolist()[0]
            self.current_state = Fsm.DOT
            return tokens

        if state == Fsm.DOT:
            tokens = np.array(self.__model.encode("0123456789")).tolist()[0]
            tokens += np.array(self.__model.encode(".")).tolist()[0]
            tokens += np.array(self.__model.encode(",")).tolist()[0]
            tokens += np.array(self.__model.encode("}")).tolist()[0]
            return tokens

        if state == Fsm.ALPHANUM:
            tokens = np.array(self.__model.encode("0123456789,}")).tolist()[0]
            return tokens

    def __convert_to_float(self)-> None:

        if "." not in self.generated:
            self.generated += "."
            self.generated += "0"

        if self.generated[-1] == ".":
            self.generated += "0"
     
    def __my_encode(self, string: str)-> List[int]:
        return np.array(self.__model.encode(string)).tolist()[0]
