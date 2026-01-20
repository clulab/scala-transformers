package org.clulab.scala_transformers.tokenizer.test

import org.clulab.transformers.test.Test

class TokenizerTest extends Test {
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")

  behavior of "Tokenizer"

  it should "tokenize with the microsoft/deberta-v3-base tokenizer" in {
    val tokenizer = new MultiTokenizer("microsoft/deberta-v3-base", addPrefixSpace = false)
    val tokenization = tokenizer.tokenize(words)

    // The special symbols are not added and words are broken up.  I think this was added later.
    val expectedTokens = Array("[CLS]", "_EU", "_rejects", "_German", "_call", "_to", "_boycott", "_British", "_lamb", "_.", "[SEP]")
        .map(word => word.replace('_', '\u2581'))
    val actualTokens = tokenization.tokens
    actualTokens should contain theSameElementsInOrderAs expectedTokens

    val expectedTokenIds = Array(1, 2805, 27144, 2324, 660, 264, 20007, 1668, 12649, 323, 2)
    val actualTokenIds = tokenization.tokenIds
    actualTokenIds should contain theSameElementsInOrderAs expectedTokenIds

    val expectedWordIds = Array(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1)
    val actualWordIds = tokenization.wordIds
    actualWordIds should contain theSameElementsInOrderAs expectedWordIds
  }
}
