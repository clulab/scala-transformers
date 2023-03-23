
from names import names
from parameters import parameters
from transformers import AutoTokenizer

class CluTokenizer:
    @classmethod
    def get_pretrained(cls, name: str = parameters.transformer_name, add_prefix_space: bool = False) -> AutoTokenizer:
        # which transformer to use
        print(f"Loading tokenizer named \"{name}\" with add_prefix_space={add_prefix_space}...")
        tokenizer = AutoTokenizer.from_pretrained(name, model_input_names=[names.INPUT_IDS, "token_type_ids", "attention_mask"])
        return tokenizer
