
from processors.core import (Parameters, Task)
from transformers import AutoTokenizer, AutoConfig
from typing import List

__all__ = ["BasicTrainer"]

# FIXME: why doesn't this inherit from Trainer?
class BasicTrainer:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.config: AutoConfig = AutoConfig.from_pretrained(Parameters.transformer_name)
        self.tokenizer: AutoTokenizer = tokenizer

    def train(self, tasks: List[Task]) -> None:
        pass
