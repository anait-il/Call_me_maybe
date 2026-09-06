from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np


class Fsm(Enum):
    START = 0
    CHAR = 1


class String:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: List[int],
                 user_prompt: str)-> None:

        self.__model: Small_LLM_Model = model 
        self.prompt: List[int] = prompt
        self.user_prompt: str = user_prompt

    def generate_string(self)-> str:

        self.__generated: str = ""
        self.__current_state: Fsm = Fsm.START
        while True:

            logits: List[int] = self.__model.get_logits_from_input_ids(self.prompt)
            masked_logits: List[int] = self.__get_masked_logits(logits)
            next_token: int = np.argmax(masked_logits)
            print(self.__model.decode([next_token]))
            next_token_decode: str = self.__model.decode([next_token])
            self.__generated += next_token_decode

            if self.__current_state == Fsm.START:
                self.__current_state = Fsm.CHAR

            elif "\"" in next_token_decode:
                index: int = next_token_decode.find("\"")
                if index == 0:
                    continue
                if self.__generated[index - 1] != "\\":
                    break 

            elif len(self.__generated) > len(self.user_prompt):
                break

        return self.__generated

    def __get_masked_logits(self, logits: List[int])-> List[int]:

        mask: List[int] = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.__current_state)
        
        if not allowed_tokens:
            return logits
        else:
            mask[allowed_tokens] = 0
            return mask + logits

    def __get_tokens(self, state: Fsm)-> List[int]:

        tokens: List[int] = []

        if state == Fsm.START:
            tokens = np.array(self.__model.encode("\"")).tolist()[0]
            return tokens

        elif state == Fsm.CHAR:
            return tokens
