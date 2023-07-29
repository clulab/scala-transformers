from clu_tokenizer import CluTokenizer
from file_utils import FileUtils
from names import Names
from parameters import Parameters

def run_document(in_file_name: str, out_file_name: str, tokenizer_name: str) -> None:
    tokenizer = CluTokenizer.get_pretrained(tokenizer_name)
    with FileUtils.for_writing(out_file_name) as out_file:
        with FileUtils.for_reading(in_file_name) as in_file:
            for line in in_file:
                line = line.strip()

                tokenized_line = tokenizer(line)
                ids_from_line = tokenized_line.input_ids
                tokens_from_line = tokenizer.convert_ids_to_tokens(ids_from_line)

                words = line.split(" ")
                tokenized_words = tokenizer(words, is_split_into_words=True)
                ids_from_words = tokenized_words.input_ids
                tokens_from_words = tokenizer.convert_ids_to_tokens(ids_from_words)

                # This never happens, probably because the sentences are pre-tokenized
                # with spaces added between the tokens.
                if tokens_from_line != tokens_from_words:
                    print(tokens_from_line, tokens_from_words)

                print(line, file=out_file)
                print(tokens_from_words, file=out_file)
                print(ids_from_words, file=out_file)

def run_directory(directory_name: str, in_document_name: str) -> None:
    in_file_name = f"{directory_name}/{in_document_name}"
    for tokenizer_name in Names.TOKENIZER_NAMES:
        out_file_name = f"{directory_name}/{Parameters.get_model_name(tokenizer_name)}.txt"
        run_document(in_file_name, out_file_name, tokenizer_name)


if __name__ == "__main__":
    run_directory("../corpora/sentences", "sentences.txt")
    run_directory("../corpora/words", "words.txt")
