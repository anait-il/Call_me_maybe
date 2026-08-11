from pathlib import Path
from parsing import ParsingContent, validate_tests, validate_def
from pydantic import ValidationError


def main():
    print("Hello from call-me-maybe!")
    try:
        validate_tests("data/input/function_calling_tests.json")
        validate_def("data/input/function_definition.json")

    except ValidationError as e:
        print(f"Error: {e.errors()[0]['msg'].strip('Value error, ')}")

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main() 
