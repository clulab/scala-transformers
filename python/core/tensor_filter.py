from torch import Tensor
from typing import Union

__all__ = ["TensorFilter"]

class TensorFilter():
    def __init__(self, predicate: Tensor) -> None:
        self.predicate: Tensor = predicate

    def filter(self, collection: Union[Tensor, None]) -> Union[Tensor, None]: # In Python 3.10 add | None
        return None if collection is None else collection[self.predicate]

    def __str__(self) -> str:
        return str(self.predicate)
