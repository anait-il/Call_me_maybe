from validation_input import Parser
from function_name import FunctionName
from function_parameters import ParametersGenerator
from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Dict, Any, List
import os
import json
import time
import rich


class Enginne():

    def __init__(self,
                 parser: Parser,
                 model: Small_LLM_Model) -> None:

        self.parser: Parser = parser
        self.__model: Small_LLM_Model = model
        self.functions_definition: List[Dict[str, Any]] = (
            self.parser.functions_definition
            )
        self.my_container: List[Dict[str, Any]] = []

    def __get_container(self) -> Dict[str, Any]:

        return {
            "prompt": None,
            "name": None,
            "parameters": None
        }

    def __encapsulation(self,
                        prompt: str,
                        function_name: str,
                        params: str) -> Dict[str, Any]:

        element: Dict[str, str] = self.__get_container()
        element["prompt"] = prompt
        element["name"] = function_name
        element["parameters"] = params
        self.my_container.append(element)

        return element

    def __generate_output_file(self, called_function: Dict[str, Any]) -> None:

        path: str = self.parser.ouput

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.parser.ouput, "+w") as f:
            json.dump(called_function, f)

    def __get_elapsed_time(self, end: float, start: float) -> None:

        elapsed_time: float = end - start
        minutes, second = divmod(elapsed_time, 60)
        rich.print("\nGeneration completed in:"
                   f"[green]{int(minutes)}m {int(second)}s[/]")

    def start_generation(self) -> None:

        start_time: float = time.perf_counter()
        user_prompt: str = ""
        for i, prompt in enumerate(self.parser.prompts):

            rich.print(f"\nProcessing [green]{i+1}[/] function[gold]...[/]\n")
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
            print(output)

        end_time: float = time.perf_counter()
        self.__get_elapsed_time(end_time, start_time)
        self.__generate_output_file(output)
