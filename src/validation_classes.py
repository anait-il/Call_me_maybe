from pydantic import BaseModel, model_validator, ConfigDict, Field
from typing import Dict, List, Any


class ParsingContent(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    prompt: str = Field(min_length=1)


class ParsingDefinition(BaseModel):
    content: Dict[str, Any]

    @model_validator(mode="after")
    def check(self) -> "ParsingDefinition":

        keys: List[str] = [
            "name",
            "description",
            "parameters",
            "returns"
        ]

        if len(self.content) < 4:
            raise ValueError("Invalide function definition: "
                             "missid argument/arguments, "
                             "expected keys"
                             "{name, description, parameters, return}")

        for key, value in self.content.items():
  
            if key.lower() not in keys:
                raise ValueError(
                    f"Invalid definition field '{key}': "
                    f"expected one of {keys}."
                )

            if not value:
                raise ValueError(
                    f"Invalid value for '{key}': the field cannot be empty."
                )

            if key.lower() in ("name", "description"):
                if not isinstance(value, str):
                    raise ValueError(
                        f"Invalid value for '{key}': "
                        f"expected a string, but got {type(value).__name__}."
                    )

            if key.lower() == "parameters":
                if not isinstance(value, dict):
                    raise ValueError(
                        "Invalid 'parameters' field: "
                        "expected an object/dictionary,"
                        f"but got {type(value).__name__}."
                    )

                self.check_parameter(value)

            if key.lower() == "returns":
                if not isinstance(value, dict):
                    raise ValueError(
                        "Invalide 'returns' field: "
                        "expecte an dictionary,"
                        f"but got {type(value).__name__}."
                    )

                self.check_return_type(value)

        return self

    def check_parameter(self, param: Dict[str, Dict[str, str]]) -> None:

        for parameter_name, parameter_definition in param.items():

            if not isinstance(parameter_name, str):
                raise ValueError(
                    "Invalid parameter definition: "
                    f"parameter name must be a string, but got "
                    f"{type(parameter_name).__name__}."
                )

            if not isinstance(parameter_definition, dict):
                raise ValueError(
                    f"Invalid definition for parameter '{parameter_name}': "
                    f"expected an object containing the parameter definition, "
                    f"but got {type(parameter_definition).__name__}."
                )

            self.check_param_type(parameter_definition, parameter_name)

    def check_param_type(self,
                         type: Dict[str, str],
                         parameter_name: str) -> None:

        allowed_types = ["number", "integer", "string", "boolean"]

        for key, value in type.items():

            if key != "type":
                raise ValueError(
                    f"Invalid definition for parameter '{parameter_name}': "
                    f"unexpected field '{key}'. "
                    "The parameter definition must contain exactly the "
                    "'type' field."
                )

            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid type for parameter '{parameter_name}': "
                    f"the 'type' value must be a string, "
                    f"but got {type(value).__name__}."
                )

            if value not in allowed_types:
                raise ValueError(
                    f"Invalid type for parameter '{parameter_name}': "
                    f"'{value}' is not a supported parameter type. "
                    f"Expected one of: {allowed_types}."
                )

    def check_return_type(self, type: Dict[str, str]) -> None:

        allowed_types = ["number", "integer", "string", "boolean"]

        for key, value in type.items():

            if key != "type":
                raise ValueError(
                    f"Invalid definition for return: "
                    f"unexpected field '{key}'. "
                    "The return definition must contain exactly the "
                    "'type' field."
                )

            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid type for return: "
                    f"the 'type' value must be a string, "
                    f"but got {type(value).__name__}."
                )

            if value not in allowed_types:
                raise ValueError(
                    f"Invalid type for return: "
                    f"'{value}' is not a supported return type. "
                    f"Expected one of: {allowed_types}."
                )
