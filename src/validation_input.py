import os
import json
from .validation_classes import ParsingContent, ParsingDefinition
from pydantic import ValidationError
from typing import Dict, Any, List, Tuple
from argparse import ArgumentParser


class Parser():
    """Parse and validate command-line input files."""

    def parsing_input_files(self) -> None:
        self.get_args()
        self.prompts: List[Dict[str, str]] = self.validate_prompt(self.input)
        self.functions_definition: List[Dict[str, Any]] = (
            self.validate_def(self.functions_definition_path))

    def get_args(self) -> None:
        """Parse and validate command-line input files."""

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
        """Load and validate the prompts from a JSON file.

        Args:
            file: Path to the prompts JSON file.

        Returns:
            A list of validated prompt dictionaries.

        Raises:
            ValueError: If the file or its contents are invalid.
        """

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
            raise ValueError(f"[ValidationError]: {e}")

        return data

    def validate_def(self, file: str) -> List[Dict[str, Any]]:
        """Load and validate function definitions from a JSON file.

        Args:
            file: Path to the function definitions JSON file.

        Returns:
            A list of validated function definition dictionaries.

        Raises:
            ValueError: If the file or its contents are invalid.
        """

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
                    "[Error] in definitions file: Invalid data (empty dict)")

        names: List[str] = []
        for content in data:

            try:
                ParsingDefinition.model_validate(content)

                if content['name'] in names:
                    raise ValueError("[ValidationError]: "
                                     "Dublicate function "
                                     f"({content['name']})")

                names.append(content['name'])

            except ValidationError as e:
                raise ValueError(f"[ValidationError]: {e.errors()[0]['msg']}")

        return data

    def my_hook(self, pairs: List[Tuple[str, str]]) -> Dict[str, str]:
        """Build a dictionary while detecting duplicate keys.

        Args:
            pairs: Key-value pairs parsed from a JSON object.

        Returns:
            A dictionary containing the parsed key-value pairs.

        Raises:
            ValueError: If duplicate keys are found.
        """

        result: Dict[str, str] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Dublicate keys '{key}'")
            result[key] = value

        return result
