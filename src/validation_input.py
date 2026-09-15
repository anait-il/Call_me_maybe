import os
import json
from .validation_classes import ParsingContent, ParsingDefinition
from pydantic import ValidationError
from typing import Dict, Any, List, Tuple
from argparse import ArgumentParser


class Parser():

    def parsing_input_files(self) -> None:
        self.get_args()
        self.prompts: List[Dict[str, str]] = self.validate_prompt(self.input)
        self.functions_definition: List[Dict[str, Any]] = (
            self.validate_def(self.functions_definition_path))

    def get_args(self) -> None:

        parser = ArgumentParser()
        parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json")
        parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json")
        parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json")

        args = parser.parse_args()

        self.input: str = args.input
        self.ouput: str = args.output
        self.functions_definition_path: str = args.functions_definition

    def validate_prompt(self, file: str) -> List[Dict[str, str]]:

        if not os.path.getsize(file):
            raise ValueError("[Error]: empty file")
        with open(file) as f:
            try:
                data = json.load(f, object_pairs_hook=self.my_hook)
            except ValueError as e:
                raise ValueError(f"[JsonError]: {e}")

        if not isinstance(data, list) and not isinstance(data, dict):
            raise ValueError("[Error]: Invalid json data.")

        data_type: str = data.__class__.__name__
        if not data:
            raise ValueError(
                f"[Error] in prompts file: Invalid data (empty {data_type})")

        if isinstance(data, dict):
            data = [data]

        try:
            for content in data:
                ParsingContent.model_validate(content)
        except ValidationError as e:
            raise ValueError("[ValidationError]: "
                    f"{e.errors()[0]['msg']}")

        return data

    def validate_def(self, file: str) -> List[Dict[str, Any]]:

        if not os.path.getsize(file):
            raise ValueError("empty file")

        with open(file) as f:

            try:
                data = json.load(f, object_pairs_hook=self.my_hook)
            except ValueError as e:
                    raise ValueError(f"[JsonError]: {e}")

        if not isinstance(data, list) and not isinstance(data, dict):
                    raise ValueError("[Error]: Invalid json data.")

        data_type: str = data.__class__.__name__
        if not data:
            raise ValueError(
                "[Error]:"
                f"Error in definitions file: Invalid data (empty {data_type})")

        if isinstance(data, dict):
            data = [data]

        for item in data:
            if not item:
                raise ValueError(
                    "Error in definitions file: Invalid data (empty dict)")

        for content in data:

            try:
                ParsingDefinition(content=content)
            except ValidationError as e:
                print("[Error]: "
                      f"{e.errors()[0]['msg'].split('Value error, ')[1]}")
                raise
            except ValueError as e:
                print(f"[Error]: {e}")
                raise

        return data

    def my_hook(self, pairs: List[Tuple[str, str]]) -> Dict[str, str]:
        print(pairs)
        result: Dict[str, str] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Dublicate keys '{key}'")
            result[key] = value
        
        return result
