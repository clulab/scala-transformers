
from parameters import parameters
from transformers import AutoTokenizer

class CluTokenizer:
    @classmethod
    def get_pretrained(cls) -> AutoTokenizer:
        # which transformer to use
        print(f'Loading tokenizer named "{parameters.transformer_name}"...')
        tokenizer = AutoTokenizer.from_pretrained(parameters.transformer_name, model_input_names=["input_ids", "token_type_ids", "attention_mask"])
        return tokenizer
