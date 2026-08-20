import os
import json
from parser_classes import ParsingContent, ParsingDefinition
from pydantic import ValidationError
from typing import Dict, Any, List
from argparse import ArgumentParser


class Parser:

    def _parse(self)-> None:

        self.get_args()
        self.prompts: List[Dict[str, str]] = self.validate_prompt(self.input)
        self.functions_definition: List[Dict[str, Any]] = self.validate_def(self.functions_definition_path)


    def get_args(self)-> None:

        parser = ArgumentParser()

        parser.add_argument("--input", default="data/input/function_calling_tests.json")
        parser.add_argument("--output", default="data/output/function_calls.json")
        parser.add_argument("--functions_definition", default="data/input/functions_definition.json")

        args = parser.parse_args()

        self.input: str = args.input
        self.ouput: str = args.output
        self.functions_definition_path: str = args.functions_definition

    def validate_prompt(self, file: str)-> List[Dict[str, str]]:

        if not os.path.getsize(file):
            raise ValueError("[Error] empty file")
        with open(file) as f:
            data = json.load(f)
            if not data:
                raise ValueError("[Error] in prompts file: Invalid data (empty list)")
            for item in data:
                if not item:
                    raise ValueError("[Error] in prompts file: Invalid data (empty dict)")

        if not isinstance(data, list):
            data = [data]

        for cotent in data:
            try:
                ParsingContent(content=cotent)

            except ValidationError as e:
                print(f"[Error] Invalide type: {e.errors()[0]['msg'].strip('Value error, ')}, "
                    "expected dict[str, str]"
            )
                raise
            except ValueError as e:
                print(e)
                raise

        return data


    def validate_def(self, file: str)-> List[Dict[str, Any]]:

        if not os.path.getsize(file):
            raise ValueError("empty file")
        with open(file) as f:
            data = json.load(f)
            if not data:
                raise ValueError("Error in definitions file: Invalid data (empty list)")
            for item in data:
                if not item:
                    raise ValueError("Error in definitions file: Invalid data (empty dict)")
        if not isinstance(data, list):
            data = [data]
        for content in data:
            try:
                ParsingDefinition(content=content)
            except ValidationError as e:
                print(f"[Error] {e.errors()[0]['msg'].split('Value error, ')[1]}")
                raise
            except ValueError as e:
                print(f"[Error] {e}")
                raise

        return data
