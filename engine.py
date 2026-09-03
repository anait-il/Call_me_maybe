from validation_input import Parser
from function_name import FunctionName
from function_parameters import ParametersGenerator
from llm_sdk import Small_LLM_Model
from typing import Dict, Any
from validation_input import Parser


class Enginne():
    def __init__(self,
                 parser: Parser,
                 model: Small_LLM_Model)-> None:

        self.parser: Parser = parser
        self.__model: Small_LLM_Model = model
        self.functions_definition: Dict[str, Any] = self.parser.functions_definition


    def __get_container(self)-> Dict[str, Any]:

        return {
            "prompt": None,
            "name": None,
            "parameters": None
        }


    def __encapsulation(self,
                        prompt: str,
                        function_name: str,
                        parameters: Dict[str, Any])-> Dict[str, Any]:

        my_container: Dict[str, Any] = self.__get_container()
        my_container["prompt"] = prompt
        my_container["name"] = function_name
        my_container["parameters"] = parameters

        return my_container


    def start_generation(self)-> None:

        for prompt in self.parser.prompts:
            name_generation = FunctionName(self.__model,
                                        prompt,
                                        self.functions_definition)
            prompt = prompt['prompt']
            print(prompt)
            name = name_generation.get_function_name(prompt)
            print(name)
            parameters_generation = ParametersGenerator(self.__model, prompt, name, self.functions_definition)
            params = parameters_generation.generate_parameter()
            output = self.__encapsulation(prompt, name, params)
            print(output)
