try:
    from .validation_input import Parser
    from llm_sdk import Small_LLM_Model  # type: ignore
    from pydantic import ValidationError
    from .engine import Enginne
except KeyboardInterrupt as e:
    print(f"Error: {e}")


def main() -> None:

    # parse prompts and functions
    parser = Parser()
    parser.parsing_input_files()

    # import llm modelform hugging face
    model = Small_LLM_Model()

    # start autoregressive  loop
    enginne = Enginne(parser, model)
    enginne.start_generation()


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, ValueError) as e:
        print(e)
        exit(1)
    except (KeyboardInterrupt) as e:
        print(f"[Error]: {e}")
