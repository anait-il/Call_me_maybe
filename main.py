from validation_input import Parser
from function_name import FunctionName
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from typing import List
import numpy as np


def main():
    print("Hello from call-me-maybe!")

    parser = Parser()
    parser._parse()

    model = Small_LLM_Model()
    function_name = FunctionName(model,
                                 parser.prompts,
                                 parser.functions_definition)
    function_name.get_function_name()


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, ValueError):
         exit(1)
