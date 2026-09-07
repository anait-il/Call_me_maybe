from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model
import numpy as np
from numpy.typing import NDArray


class Fsm(Enum):
    DIGITS = 1
    ALPHANUM = 2
    END = 3


class Integer:

    def __init__(self,
                 model: Small_LLM_Model,
                 builded_prompt: List[int],
                 user_prompt: str) -> None:

        self.__model: Small_LLM_Model = model
        self.builded_prompt: List[int] = builded_prompt
        self.user_prompt: str = user_prompt

    def generate_integer(self) -> str:

        self.__generated: str = ""
        self.__generated_tokens: List[int] = []
        self.__current_state: Fsm = Fsm.DIGITS
        while not self.__current_state == Fsm.END:

            if len(self.__generated) > len(self.user_prompt):
                break

            logits: List[float] = self.__model.get_logits_from_input_ids(
                self.builded_prompt + self.__generated_tokens)
            masked_logits: NDArray = self.__get_masked_logits(logits)
            next_token: int = int(np.argmax(masked_logits))

            if self.__current_state == Fsm.DIGITS:
                self.__current_state = Fsm.ALPHANUM

            elif self.__current_state == Fsm.ALPHANUM:
                if next_token in self.__my_encode(",}"):
                    self.__current_state = Fsm.END
                    break

            self.__generated += self.__model.decode(next_token)
            self.__generated_tokens.append(next_token)

        return self.__generated

    def __get_masked_logits(self, logits: List[float]) -> NDArray:

        mask: NDArray = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.__current_state)
        mask[allowed_tokens] = 0

        return mask + logits

    def __get_tokens(self, state: Fsm) -> List[int]:

        tokens: List[int] = []

        if state == Fsm.DIGITS:
            tokens = np.array(self.__model.encode("-0123456789")).tolist()[0]
            return tokens

        tokens = np.array(self.__model.encode("0123456789,}")).tolist()[0]
        return tokens

    def __my_encode(self, string: str) -> List[int]:

        return [int(token)
                for token in np.array(self.__model.encode(string))[0]]
