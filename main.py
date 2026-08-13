from pathlib import Path
from validation_input import validate_prompt, validate_def
from pydantic import ValidationError


def main():
    print("Hello from call-me-maybe!")
    try:
        validate_prompt("data/input/function_calling_tests.json")
        validate_def("data/input/functions_definition.json")

    except (ValueError, ValidationError):
        pass


if __name__ == "__main__":
    main() 
