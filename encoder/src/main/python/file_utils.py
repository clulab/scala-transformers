
from io import TextIOWrapper
from parameters import Parameters

class FileUtils:
    @classmethod
    def for_writing(cls, name: str) -> TextIOWrapper:
        return open(name, "w", encoding=Parameters.encoding)

    @classmethod
    def for_reading(cls, name: str) -> TextIOWrapper:
        return open(name, "r", encoding=Parameters.encoding)
