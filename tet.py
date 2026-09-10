from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np

model = Small_LLM_Model()

user_prompt = "Replace all vowels in 'Programming is fun' with asterisks"
function_definition = {
    "fn_substitute_string_with_regex": {
      "parameters": 
      {
       "source_string": "string",
       "regex": "string",
       "replecement": "string"
    }
    }
}

function_name = "fn_substitute_string_with_regex"

def main():

    prompt = build_prompt()
    tokens = model.encode(prompt).tolist()[0]
    output = ""
    while not output or "}" not in output:
        logits = model.get_logits_from_input_ids(tokens)
        # max_score = max(logits)
        # next_token = (int(x) for x, s in enumerate(logits) if s == max_score)
        next_token = np.argmax(logits)
        next_element = model.decode(next_token)
        print(next_element)
        output += next_element
        tokens.append(next_token)

    print(f"final output ->> {output}")


def build_prompt() -> str:
   
    return f"""
            You are a function-calling assistant that
            helps me get a JSON format from a user prompt.

            Available functions:
            {function_definition}

            Example:

            Prompt: "what is the sum of 1 and 2"

            Answer:
            {{
                "prompt": "what is the sum of 1 and 2",
                "name": "fn_add_numbers",
                "parameters": {{"a": 1.0, "b": 2.0}}
            }}

            User prompt: {user_prompt}

            JSON:
            {{
                "prompt": {user_prompt},
                "name": "'''{function_name}
                "parameters": 
        """


if __name__ == "__main__":
    main()
