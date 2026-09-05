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
                 prompt: List[int],
                 user_prompt: str)-> None:

        self.__model: Small_LLM_Model = model 
        self.prompt: List[int] = prompt
        self.user_prompt: str = user_prompt

    def generate_bool(self)-> str:

        self.generated: str = ""
        self.current_state: Fsm = Fsm.DIGITS
        while not self.current_state == Fsm.END:
 
            if len(self.generated) > len (self.user_prompt):
                break

            logits: List[int] = self.__model.get_logits_from_input_ids(self.prompt) 
            masked_logits: List[int] = self.__get_masked_logits(logits)
            next_token: int = np.argmax(masked_logits)
            self.generated += self.__model.decode(next_token)
            self.prompt.append(next_token)
            if self.generated == 'true' or \
                self.generated == "false":
                break

        return self.generated

    def __get_masked_logits(self, logits: List[int])-> List[int]:

        mask: List[int] = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.current_state)
        mask[allowed_tokens] = 0
 
        return mask + logits

    def __get_tokens(self, state: Fsm)-> List[int]:

        tokens: List[int] = []

        if state == Fsm.DIGITS:
            tokens = np.array(self.__model.encode("truefalse")).tolist()[0]
            self.current_state = Fsm.END
            return tokens
