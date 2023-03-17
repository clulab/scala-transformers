
from io import TextIOWrapper
from parameters import parameters

class FileUtils:
    @classmethod
    def for_writing(name: str) -> TextIOWrapper:
        return open(name, "w", encoding=parameters.encoding)

    @classmethod
    def for_reading(name: str) -> TextIOWrapper:
        return open(name, "r", encoding=parameters.encoding)
