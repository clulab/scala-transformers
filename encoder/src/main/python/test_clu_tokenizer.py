
from clu_tokenizer import CluTokenizer

def test_get_pretrained() -> None:
    tokenizer = CluTokenizer.get_pretrained()

    tokenizer = CluTokenizer.get_pretrained("bert-base-cased")
    tokenizer = CluTokenizer.get_pretrained("distilbert-base-cased")
    tokenizer = CluTokenizer.get_pretrained("roberta-base")
    tokenizer = CluTokenizer.get_pretrained("xlm-roberta-base")
    tokenizer = CluTokenizer.get_pretrained("google/bert_uncased_L-4_H-512_A-8")
    tokenizer = CluTokenizer.get_pretrained("google/electra-small-discriminator")
    tokenizer = CluTokenizer.get_pretrained("microsoft/deberta-v3-base")

if __name__ == "__main__":
    test_get_pretrained()
