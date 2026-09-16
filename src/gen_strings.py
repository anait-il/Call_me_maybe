from enum import Enum
from typing import List
from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from numpy.typing import NDArray


class Fsm(Enum):
    """Represent the states of string value generation."""

    START = 0
    CHAR = 1
    END = 2


class String:
    """Generate a string value using constrained decoding."""

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: List[int],
                 user_prompt: str) -> None:
        """Initialize the string value generator.

        Args:
            model: Language model used for generation.
            prompt: Tokenized prompt used as model input.
            user_prompt: Original user request.
        """

        self.__model: Small_LLM_Model = model
        self.prompt: List[int] = prompt
        self.user_prompt: str = user_prompt

    def generate_string(self) -> str:
        """Generate a JSON-compatible string value.

        Returns:
            The generated string value.
        """

        self.__generated: str = ""
        self.__generated_tokens: List[int] = []
        self.__current_state: Fsm = Fsm.START
        escaped = {
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
            "\b": "\\b",
            "\f": "\\f",
            }

        while self.__current_state != Fsm.END:

            logits: List[float] = self.__model.get_logits_from_input_ids(
                self.prompt + self.__generated_tokens)
            masked_logits: List[float] | NDArray = (
                self.__get_masked_logits(logits))
            next_token: int = int(np.argmax(masked_logits))

            next_token_decode: str = self.__model.decode([next_token])

            for c in escaped:
                if c in next_token_decode:
                    next_token_decode = (
                        next_token_decode.replace(c, escaped[c]))

            if "\"" in next_token_decode and self.__current_state == Fsm.CHAR:

                if self.__is_there_backslash(
                        self.__generated + next_token_decode):
                    self.__current_state = Fsm.END

                    next_token_decode = next_token_decode.split('"')[0] + "\""

            elif "\\" in next_token_decode:
                next_token_decode = next_token_decode.replace("\\", "\\\\")

            elif self.__current_state == Fsm.START:
                self.__current_state = Fsm.CHAR

            elif len(self.__generated) > len(self.user_prompt):
                self.__current_state = Fsm.END

            self.__generated += next_token_decode
            print(next_token_decode)
            self.__generated_tokens.append(next_token)

        return self.__generated

    def __get_masked_logits(self,
                            logits: List[float]) -> List[float] | NDArray:
        """Mask logits to allow only valid string tokens.

        Args:
            logits: Logits produced by the language model.

        Returns:
            Logits with invalid tokens masked.
        """

        mask: NDArray = np.full_like(logits, float("-inf"))
        allowed_tokens: List[int] = self.__get_tokens(self.__current_state)

        if not allowed_tokens:
            return logits
        else:
            mask[allowed_tokens] = 0
            return mask + logits

    def __get_tokens(self, state: Fsm) -> List[int]:
        """Get token IDs allowed in the current FSM state.

        Args:
            state: Current state of string generation.

        Returns:
            Token IDs allowed for the current state.
        """

        tokens: List[int] = []

        if state == Fsm.START:
            tokens = np.array(self.__model.encode("\"")).tolist()[0]
            return tokens

        return tokens

    def __is_there_backslash(self, output: str) -> bool:
        """Check whether a quote is preceded by an even number of backslashes.

        Args:
            output: Generated string to inspect.

        Returns:
            True if the quote is not escaped, otherwise False.
        """

        index: int = output.rfind("\"")
        number_of_backslashes: int = 0
        x: int = 1

        while index - x > 0 and output[index - x] == "\\":
            number_of_backslashes += 1
            x += 1

        return (number_of_backslashes % 2 == 0)
