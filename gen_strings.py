from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np


class Fsm(Enum):
    START = 0
    CHAR = 1
    END = 2


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
        self.__generated_tokens: List[int] = []
        self.__current_state: Fsm = Fsm.START

        while self.__current_state != Fsm.END:

            logits: List[int] = self.__model.get_logits_from_input_ids(
                self.prompt + self.__generated_tokens)
            masked_logits: List[int] = self.__get_masked_logits(logits)
            next_token: int = np.argmax(masked_logits)

            next_token_decode: str = self.__model.decode([next_token])

            if "\"" in next_token_decode and self.__current_state == Fsm.CHAR:
                index: int = next_token_decode.find("\"")
                if self.__generated[index - 1] != "\\":
                    self.__current_state = Fsm.END
                    next_token_decode = "\""

            elif self.__current_state == Fsm.START:
                self.__current_state = Fsm.CHAR

            elif len(self.__generated) > len(self.user_prompt):
                self.__current_state = Fsm.END

            self.__generated += next_token_decode
            self.__generated_tokens.append(next_token)

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
            tokens = np.array(self.__model.encode(" \"")).tolist()[0]
            return tokens

        elif state == Fsm.CHAR:
            return tokens
