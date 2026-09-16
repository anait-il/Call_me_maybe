from .validation_input import Parser
from .function_name import FunctionName
from .function_parameters import ParametersGenerator
from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Dict, Any, List
import os
import json
from json.decoder import JSONDecodeError
import time
import rich


class Enginne():
    """Coordinate function selection, parameter generation, and output."""

    def __init__(self,
                 parser: Parser,
                 model: Small_LLM_Model) -> None:
        """Initialize the generation engine.

        Args:
            parser: Parser containing validated input data.
            model: Language model used for generation.
        """

        self.parser: Parser = parser
        self.__model: Small_LLM_Model = model
        self.functions_definition: List[Dict[str, Any]] = (
            self.parser.functions_definition
            )

    def __get_container(self) -> Dict[str, Any]:
        """Create an empty function-call result container.

        Returns:
            A dictionary containing the prompt, function name, \
            and parametr keys.
        """

        return {
            "prompt": None,
            "name": None,
            "parameters": None
        }

    def __encapsulation(self,
                        prompt: str,
                        function_name: str,
                        params: str) -> Dict[str, Any]:
        """Build a function-call result from generated values.

        Args:
            prompt: User prompt associated with the function call.
            function_name: Name of the selected function.
            params: JSON string containing generated parameters.

        Returns:
            A dictionary containing the prompt, function name, and parameters.

        Raises:
            ValueError: If the generated parameters are invalid JSON.
        """

        element: Dict[str, Any] = self.__get_container()
        element["prompt"] = prompt
        element["name"] = function_name
        try:
            element["parameters"] = json.loads(params)
        except JSONDecodeError:
            raise ValueError(f"[JsonError] LLM generated invalid json:"
                             f"{params}")

        return element

    def __generate_output_file(self,
                               called_function: List[Dict[str, Any]]) -> None:
        """Write generated function calls to the output JSON file.

        Args:
            called_function: List of generated function-call results.
        """

        path: str = self.parser.ouput

        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(self.parser.ouput, "w") as f:
                json.dump(called_function, f, indent=4)
        except OSError as e:
            raise OSError(f"[FileError]: {e}")

    def __get_elapsed_time(self, end: float, start: float) -> None:
        """Display the elapsed generation time.

        Args:
            end: End time measured by the performance timer.
            start: Start time measured by the performance timer.
        """

        elapsed_time: float = end - start
        minutes, second = divmod(elapsed_time, 60)
        rich.print("\nGeneration completed in:"
                   f"[green]{int(minutes)}m {int(second)}s[/]")

    def start_generation(self) -> None:
        """Generate function calls for all validated prompts."""

        start_time: float = time.perf_counter()
        user_prompt: str = ""
        generated: List[Dict[str, Any]] = []
        for i, prompt in enumerate(self.parser.prompts):

            rich.print(f"\nFunction [green]{i+1}[/] processing[gold]...[/]\n")
            user_prompt = prompt['prompt']
            name_generation: FunctionName = FunctionName(
                self.__model,
                user_prompt,
                self.functions_definition)
            func_name: str = name_generation.generate_function_name()
            parameters_generation: ParametersGenerator = ParametersGenerator(
                self.__model,
                user_prompt,
                func_name,
                self.functions_definition)
            params: str = parameters_generation.generate_parameter()
            output = self.__encapsulation(user_prompt, func_name, params)
            generated.append(output)
            print(output)

        end_time: float = time.perf_counter()
        self.__get_elapsed_time(end_time, start_time)
        self.__generate_output_file(generated)
