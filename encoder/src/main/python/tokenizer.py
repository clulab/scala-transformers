import configuration as cf

from transformers import AutoTokenizer

class Tokenizer():

    @classmethod
    def get_pretrained(cls) -> AutoTokenizer:
        # which transformer to use
        print(f'Loading tokenizer named "{cf.transformer_name}"...')
        tokenizer = AutoTokenizer.from_pretrained(cf.transformer_name, model_input_names=["input_ids", "token_type_ids", "attention_mask"])
        return tokenizer