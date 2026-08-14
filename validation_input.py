import os
import json
from parser_classes import ParsingContent, ParsingDefinition
from pydantic import ValidationError


def validate_prompt(file: str)-> None:
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


def validate_def(file: str)-> None:
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