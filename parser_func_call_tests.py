from types import Dict
from enum import Enum

class File_type(Enum):
    TESTS = "tests"
    DEFINITION = "definition"


class Parsing:
    def __init__(self, type: File_type , data):
        self.type = type
        self.data = data

    def parse_func_call_tests(data: Dict[str, str]):
        for element in data:
            counter = 0
            for key, value in element:
                counter += 1
                if not key == "prompt":
                    raise ValueError("Error: Invalid key in function_calling_tests.py")
                if not value:
                    raise ValueError("Error Invalid prompt: Can't be empty")
