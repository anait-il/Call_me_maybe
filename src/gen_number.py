from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from numpy.typing import NDArray


class Fsm(Enum):
    """Represent the states of number value generation."""

    DIGITS = 1
    ALPHANUM = 2
    END = 3


class Number:
    """Generate a number value using constrained decoding."""

    def __init__(self,
                 model: Small_LLM_Model,
                 building_prompt: List[int],
                 user_prompt: str) -> None:
        """Initialize the number value generator.

        Args:
            model: Language model used for generation.
            building_prompt: Tokenized prompt used as model input.
            user_prompt: Original user request.
        """

        self.__model: Small_LLM_Model = model
        self.building_prompt: List[int] = building_prompt
        self.user_prompt: str = user_prompt

    def generate_numbers(self) -> str:
        """Generate a number value.

        Returns:
            The generated number value.
        """

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
        """Mask logits to allow only valid number tokens.

        Args:
            logits: Logits produced by the language model.

        Returns:
            Logits with invalid tokens masked.
        """

        mask: NDArray = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.__current_state)
        mask[allowed_tokens] = 0

        return mask + logits

    def __get_tokens(self, state: Fsm) -> List[int]:
        """Get token IDs allowed in the current FSM state.

        Args:
            state: Current state of number generation.

        Returns:
            Token IDs allowed for the current state.
        """

        tokens: List[int] = []

        if state == Fsm.DIGITS:
            for token in "-0123456789":

                tokens += np.array(self.__model.encode(token)).tolist()[0]

            return tokens

        for token in "0123456789.,}":

            tokens += np.array(self.__model.encode(token)).tolist()[0]

        return tokens

    def __convert_to_float(self) -> None:
        """Ensure the generated number has a decimal part."""

        if "." not in self.__generated:
            self.__generated += "."
            self.__generated += "0"

        if self.__generated[-1] == ".":
            self.__generated += "0"

    def __my_encode(self, string: str) -> List[int]:
        """Encode a string into token IDs.

        Args:
            string: Text to encode.

        Returns:
            Token IDs corresponding to the input string.
        """

        return [int(token)
                for token in np.array(self.__model.encode(string))[0]]
