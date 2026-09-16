from pydantic import BaseModel, model_validator, ConfigDict, Field
from typing import Dict
import keyword


class ParsingContent(BaseModel):
    """Validate a prompt input object."""

    model_config = ConfigDict(
        extra="forbid"
    )

    prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_prompt(self) -> "ParsingContent":
        """Validate that the prompt is not empty or whitespace-only."""

        if not self.prompt.strip():
            raise ValueError("Invalid prompt (empty)")

        return self


class ParsingDefinition(BaseModel):
    """Validate a function definition and its parameter types."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]

    @model_validator(mode="after")
    def check(self) -> "ParsingDefinition":
        """Validate the function name, description, parameters, \
            and return type."""

        if not self.name.isidentifier() or keyword.iskeyword(self.name):
            raise ValueError(f"Invalid function name ({self.name})")

        if not self.description.strip():
            raise ValueError("Invalid Description (empty)")

        for key, value in self.parameters.items():

            if not key.isidentifier() or keyword.iskeyword(key):
                raise ValueError(f"Invalid parameter name ({key})")

            if not value:
                raise ValueError("Invalid parameter type (empty)")

            self.check_param_type(value, key)

        self.check_return_type(self.returns)

        return self

    def check_param_type(self,
                         parameter: Dict[str, str],
                         parameter_name: str) -> None:
        """Validate a parameter's type definition.

        Args:
            parameter: Parameter type definition to validate.
            parameter_name: Name of the parameter being validated.

        Raises:
            ValueError: If the definition or type is invalid.
        """

        allowed_types = ["number", "integer", "string", "boolean"]

        key, value = next(iter(parameter.items()))

        if key != "type":
            raise ValueError(
                f"Invalid definition for parameter '{parameter_name}': "
                f"unexpected field '{key}'. "
                "The parameter definition must contain exactly the "
                "'type' field."
            )

        if value not in allowed_types:
            raise ValueError(
                f"Invalid type for parameter '{parameter_name}': "
                f"'{value}' is not a supported parameter type. "
                f"Expected one of: {allowed_types}.")

    def check_return_type(self, returns: Dict[str, str]) -> None:
        """Validate the function's return type definition.

        Args:
            returns: Return type definition to validate.

        Raises:
            ValueError: If the definition or type is invalid.
        """

        allowed_types = ["number", "integer", "string", "boolean"]

        if not returns:
            return

        key, value = next(iter(returns.items()))
        if key != "type":
            raise ValueError(
                f"Invalid definition for returns: "
                f"unexpected field '{key}'. "
                "The return definition must contain exactly the "
                "'type' field."
            )

        if value not in allowed_types:
            raise ValueError(
                f"Invalid type for return: "
                f"'{value}' is not a supported return type. "
                f"Expected one of: {allowed_types}."
            )
