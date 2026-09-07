from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np
from numpy.typing import NDArray


class Fsm(Enum):
    DIGITS = 0
    END = 1


class Boolean:

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: List[int],
                 user_prompt: str) -> None:

        self.__model: Small_LLM_Model = model
        self.prompt: List[int] = prompt
        self.user_prompt: str = user_prompt

    def generate_bool(self) -> str:

        self.__generated: str = ""
        self.__current_state: Fsm = Fsm.DIGITS
        while not self.__current_state == Fsm.END:

            if len(self.__generated) > len("false"):
                break

            logits: List[float] = (
                self.__model.get_logits_from_input_ids(self.prompt))
            masked_logits: NDArray = self.__get_masked_logits(logits)
            next_token: int = int(np.argmax(masked_logits))
            self.__generated += self.__model.decode(next_token)
            self.prompt.append(next_token)
            if self.__generated.lower() == 'true' or \
                self.__generated.lower() == "false" or \
                self.__generated == "0" or \
               self.__generated == "1":
                break

        return self.__generated

    def __get_masked_logits(self, logits: List[float]) -> NDArray:

        mask: NDArray = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.__current_state)
        mask[allowed_tokens] = 0

        return mask + logits

    def __get_tokens(self, state: Fsm) -> List[int]:

        tokens: List[int] = []

        if state == Fsm.DIGITS:
            tokens = np.array(self.__model.encode("truefalse")).tolist()[0]
            tokens += np.array(self.__model.encode("TrueFalse")).tolist()[0]
            tokens += np.array(self.__model.encode("TRUEFALSE")).tolist()[0]
            tokens += np.array(self.__model.encode("01")).tolist()[0]
            self.__current_state = Fsm.END
            return tokens

        return tokens
