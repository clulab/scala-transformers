
from parameters import parameters
from task import Task
from transformers import AutoTokenizer, AutoConfig
from typing import List

class BasicTrainer:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.config: AutoConfig = AutoConfig.from_pretrained(parameters.transformer_name)
        self.tokenizer: AutoTokenizer = tokenizer

    def train(self, tasks: List[Task]) -> None:
        pass
