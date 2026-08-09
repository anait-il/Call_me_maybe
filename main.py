from pathlib import Path
import json
import sys
#import parser_func_call_tests

def main():
    print("Hello from call-me-maybe!")
    with open("data/input/function_calling_tests.json") as tests:
        function_calling_data = json.load(tests)

    print(type(function_calling_data))
    for element in function_calling_data:
        print(element)

if __name__ == "__main__":
    main() 
