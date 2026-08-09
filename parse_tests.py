from pydantic import BaseModel
from types import Dict
from enum import Enum

class FileType(Enum):
    TESTS = "tests"
    DEFINITION = "definition"


class Parsing(BaseModel):
    
