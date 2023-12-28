
from processors.core import (Names, Parameters)
from transformers import AutoTokenizer

__all__ = ["CluTokenizer"]

class CluTokenizer:
    
    # FIXME: set use_fast based on availability of fast tokenizer
    @classmethod
    def from_pretrained(cls, name: str = Parameters.transformer_name, use_fast: bool = True) -> AutoTokenizer:
        # add_prefix_space is needed only for the roberta tokenizer
        add_prefix_space = "roberta" in name.lower()
        # which transformer to use
        print(f"Loading tokenizer named \"{name}\" with add_prefix_space={add_prefix_space}...")
        tokenizer = AutoTokenizer.from_pretrained(name, model_input_names=[Names.INPUT_IDS, "token_type_ids", "attention_mask"], add_prefix_space=add_prefix_space, use_fast=use_fast)
        return tokenizer
