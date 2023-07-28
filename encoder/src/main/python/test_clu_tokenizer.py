
from clu_tokenizer import CluTokenizer
from names import Names

def test_clu_tokenizer() -> None:
    words = [
        "xkcd", "supercalifragilisticexpialidocious", "opyt",
        "<s>", "[CLS]", "[SEP]", "[UNK]", "\u2581", "\u0120",
        "a", "b", "c", "d",
        "\u1000", "\u2000", "\u3000", "\u4000", "\ue000", "\uf000"
    ]


    tokenizer = CluTokenizer.get_pretrained()
    for name in Names.TOKENIZER_NAMES:
        tokenizer = CluTokenizer.get_pretrained(name)
        tokenized_words = tokenizer(words, is_split_into_words=True)
        ids_from_words = tokenized_words.input_ids
        tokens_from_words = tokenizer.convert_ids_to_tokens(ids_from_words)

        print(words)
        print(tokens_from_words)
        print(ids_from_words)


if __name__ == "__main__":
    test_clu_tokenizer()
