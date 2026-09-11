from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from numpy.typing import NDArray


class Fsm(Enum):
    DIGITS = 1
    ALPHANUM = 2
    END = 3


class Number:

    def __init__(self,
                 model: Small_LLM_Model,
                 building_prompt: List[int],
                 user_prompt: str) -> None:

        self.__model: Small_LLM_Model = model
        self.building_prompt: List[int] = building_prompt
        self.user_prompt: str = user_prompt

    def generate_numbers(self) -> str:

        self.__generated: str = ""
        self.__generated_tokens: List[int] = []
        self.__current_state: Fsm = Fsm.DIGITS
        while not self.__current_state == Fsm.END:

            if len(self.__generated) > len(self.user_prompt):
                break
            logits: List[float] = self.__model.get_logits_from_input_ids(
                (self.building_prompt + self.__generated_tokens))
            masked_logits: NDArray = self.__get_masked_logits(logits)
            next_token: int = int(np.argmax(masked_logits))
            if self.__current_state == Fsm.DIGITS:
                self.__current_state = Fsm.ALPHANUM

            elif next_token == self.__my_encode("."):
                if self.__generated.find(".") != -1:
                    self.__current_state = Fsm.END
                    break

            elif next_token in self.__my_encode(",}"):
                self.__current_state = Fsm.END
                break

            self.__generated += self.__model.decode(next_token)
            self.__generated_tokens += [next_token]

        self.__convert_to_float()
        return self.__generated

    def __get_masked_logits(self, logits: List[float]) -> NDArray:

        mask: NDArray = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.__current_state)
        mask[allowed_tokens] = 0

        return mask + logits

    def __get_tokens(self, state: Fsm) -> List[int]:

        tokens: List[int] = []

        if state == Fsm.DIGITS:
            for token in "-0123456789":
                tokens += np.array(self.__model.encode(token)).tolist()[0]
            return tokens

        for token in "0123456789.,}":

            tokens += np.array(self.__model.encode(token)).tolist()[0]
        return tokens

    def __convert_to_float(self) -> None:
        if "." not in self.__generated:
            self.__generated += "."
            self.__generated += "0"

        if self.__generated[-1] == ".":
            self.__generated += "0"

    def __my_encode(self, string: str) -> List[int]:
        return [int(token)
                for token in np.array(self.__model.encode(string))[0]]
