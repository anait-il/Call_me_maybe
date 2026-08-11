from pydantic import BaseModel, ConfigDict, model_validator
from typing import Dict, List, Any
from enum import Enum
import os
import json


class ParsingContent(BaseModel):
    content: Dict[str, str]

    @model_validator(mode="after")
    def check(self):
        for key, value in self.content.items():
            if key != 'prompt':
                raise ValueError(f"Error in test file: Invalid Key {key}")

            if not isinstance(value, str):
                raise ValueError("Error in test file: The value must be string")

            if not value.strip():
                raise ValueError("Error in test file: They must not be empty")
        return self


class ParsingDefinition(BaseModel):
    content: Dict[str, str]

    @model_validator(mode="after")
    def check(self):
        keys: List[str] = ["name", "description", "parameters", "returns"]
        for key, value in self.content.items():
            if key.lower() not in keys:
                raise ValueError(f"Error: Invalide key '{key}'")
            if not value:
                raise ValueError("Error: empty value")

            if key.lower() == "name" or key.lower() == "description":
                if not isinstance(value, str):
                    raise ValueError(f"Error Invalide type for {key}: expected string got '{value}'")

            if key.lower() == "parameters":
                if not isinstance(value, Dict):
                    raise ValueError("Error: parameters must be a dict")
                self.check_parameter(value)


    def check_parameter(param: Dict[str, Dict[str, str]]) -> int:
        for key, value in param:
            types: List[str] = ["number", "integer", "string", "bool"]
            if not isinstance(key, str):
                raise ValueError(f"Error: Invalid parameter '{key}'")
            if not isinstance(value, dict):
                raise ValueError(f"Error: invalide paramter value (must be dict)")
            for k, v in value:
                if not isinstance(k, str) or k != "type":
                    raise ValueError("Error: the definition key of parameter must be exactly 'type'")
                if not isinstance(v, str):
                    raise ValueError("Error: the type must be string")
                if v not in types:
                    raise ValueError(f"Error: Invalid parameter type 'v'")


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

    for cotent in data:
        ParsingDefinition(content=cotent)


def validate_tests(file: str)-> None:

    if not os.path.getsize(file):
        raise ValueError("empty file")
    with open(file) as f:
        data = json.load(f)
        if not data:
            raise ValueError("Error in test file: Invalid data (empty list)")
        for item in data:
            if not item:
                raise ValueError("Error in test filse: Invalid data (empty dict)")

    if not isinstance(data, list):
        data = [data]

    for cotent in data:
        ParsingContent(content=cotent)
