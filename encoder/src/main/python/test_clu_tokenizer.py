
from clu_tokenizer import CluTokenizer
from names import names

def test_clu_tokenizer() -> None:
    words = [
        "xkcd", "supercalifragilisticexpialidocious", "opyt",
        "<s>", "[CLS]", "[SEP]", "[UNK]", "\u2581", "\u0120",
        "a", "b", "c", "d",
        "\u1000", "\u2000", "\u3000", "\u4000", "\ue000", "\uf000"
    ]


    tokenizer = CluTokenizer.get_pretrained()
    for name in names.tokenizer_names:
        tokenizer = CluTokenizer.get_pretrained(name)
        tokenizedWords = tokenizer(words, is_split_into_words=True)
        idsFromWords = tokenizedWords.input_ids
        tokensFromWords = tokenizer.convert_ids_to_tokens(idsFromWords)

        print(words)
        print(tokensFromWords)
        print(idsFromWords)


if __name__ == "__main__":
    test_clu_tokenizer()
