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
                 builded_prompt: List[int],
                 user_prompt: str)-> None:

        self.__model: Small_LLM_Model = model
        self.builded_prompt: List[int] = builded_prompt
        self.user_prompt: str = user_prompt

    def generate_integer(self)-> str:

        self.generated: str = ""
        self.current_state: Fsm = Fsm.SIGN
        while not self.current_state == Fsm.END:

            if len(self.generated) > len(self.user_prompt):
                break

            logits: List[int] = self.__model.get_logits_from_input_ids(self.builded_prompt) 
            masked_logits: List[int] = self.__get_masked_logits(logits)
            next_token: int = np.argmax(masked_logits)

            if self.current_state == Fsm.SIGN:
                self.current_state = Fsm.DIGITS
                if self.__model.decode([next_token]) == "+":
                    continue
    
            elif self.current_state == Fsm.DIGITS:
                self.current_state = Fsm.ALPHANUM
                
            elif self.current_state == Fsm.ALPHANUM:
                if next_token in self.__my_encode(",}"):
                    self.current_state = Fsm.END
                    break

            self.generated += self.__model.decode(next_token)
            self.builded_prompt.append(next_token)

        return self.generated

    def __get_masked_logits(self, logits: List[int])-> List[int]:

        mask: List[int] = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.current_state)
        mask[allowed_tokens] = 0
        
        return mask + logits
        
    def __get_tokens(self, state: Fsm)-> List[int]:

        tokens: List[int] = []

        if state == Fsm.SIGN:
            tokens = np.array(self.__model.encode("-+")).tolist()[0]
            return tokens

        if state == Fsm.DIGITS:
            tokens = np.array(self.__model.encode("0123456789")).tolist()[0]
            return tokens

        if state == Fsm.ALPHANUM:
            tokens = np.array(self.__model.encode("0123456789,}")).tolist()[0]
            return tokens

    def __my_encode(self, string: str)-> List[int]: 

        return np.array(self.__model.encode(string)).tolist()[0] 
