from validation_input import Parser
from function_name import FunctionName
from function_parameters import ParametersGenerator
from llm_sdk import Small_LLM_Model
from typing import Dict, Any, List


class Enginne():
    def __init__(self,
                 parser: Parser,
                 model: Small_LLM_Model) -> None:

        self.parser: Parser = parser
        self.__model: Small_LLM_Model = model
        self.functions_definition: List[Dict[str, Any]] = (
            self.parser.functions_definition
            )

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

        my_container: Dict[str, Any] = self.__get_container()
        my_container["prompt"] = prompt
        my_container["name"] = function_name
        my_container["parameters"] = params

        return my_container

    def start_generation(self) -> None:

        for prompt in self.parser.prompts:

            prompt = prompt['prompt']

            name_generation = FunctionName(
                self.__model,
                prompt,
                self.functions_definition)
            func_name = name_generation.generate_function_name()
            parameters_generation = ParametersGenerator(
                self.__model,
                prompt,
                func_name,
                self.functions_definition)
            params = parameters_generation.generate_parameter()

            output = self.__encapsulation(prompt, func_name, params)
            print(output)
            print("\n\n")
