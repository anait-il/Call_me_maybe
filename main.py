from validation_input import Parser
from function_name import FunctionName
from parameters_generator import Parameters
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from typing import Dict, Any
import numpy as np


def main():
    print("Hello from call-me-maybe!")

    parser = Parser()
    parser._parse()

    model = Small_LLM_Model()
    assembler: Dict[str, Any] = {
        "name": None,
        "parameters": None
    }

    for prompt in parser.prompts:
        function_name = FunctionName(model,
                                     prompt,
                                     parser.functions_definition)
        name = function_name.get_function_name(prompt)
        parameters = Parameters(model, prompt, name, parser.functions_definition)
        params = parameters.generate_parameter()
        assembler["name"] = name
        assembler["parameters"] = params
        print("#" * 50)
        print(assembler)


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, ValueError):
         exit(1)
