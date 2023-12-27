from processors.tokenizers import CluTokenizer
from processors.core import Names

if __name__ == "__main__":
    for name in Names.TOKENIZER_NAMES:
        # It is important to use the fast version.  Luckily, use_fast=True is the
        # default so that the code in clu_tokenizer can be reused.
        tokenizer = CluTokenizer.from_pretrained(name)
        print(tokenizer.is_fast)
        # FIXME: use pathlib
        tokenizer.save_pretrained("../pretrained/" + name)
