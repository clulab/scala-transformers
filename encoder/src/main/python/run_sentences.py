from clu_tokenizer import CluTokenizer
from file_utils import FileUtils
from parameters import parameters

if __name__ == "__main__":
    inFilename = "../corpora/sentences/sentences.txt"
    outFilename = "../corpora/sentences/" + parameters.model_name + ".txt"
    tokenizer = CluTokenizer.get_pretrained()
    with FileUtils.for_writing(outFilename) as outFile:
        with FileUtils.for_reading(inFilename) as inFile:
            for line in inFile:
                line = line.strip()

                tokenizedLine = tokenizer(line)
                idsFromLine = tokenizedLine.input_ids
                tokensFromLine = tokenizer.convert_ids_to_tokens(idsFromLine)

                words = line.split(" ")
                tokenizedWords = tokenizer(words, is_split_into_words=True)
                idsFromWords = tokenizedWords.input_ids
                tokensFromWords = tokenizer.convert_ids_to_tokens(idsFromWords)

                # This never happens, probably because the sentences are pre-tokenized
                # with spaces added between the tokens.
                if tokensFromLine != tokensFromWords:
                    print(tokensFromLine, tokensFromWords)

                print(line, file=outFile)
                print(tokensFromWords, file=outFile)
                print(idsFromWords, file=outFile)
