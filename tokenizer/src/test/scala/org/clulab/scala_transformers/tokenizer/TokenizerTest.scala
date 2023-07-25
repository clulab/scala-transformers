package org.clulab.scala_transformers.tokenizer

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.transformers.test.Test

class TokenizerTest extends Test {
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")

  behavior of "Tokenizer"

  it should "tokenize with the distilbert-base-cased tokenizer" in {
    val tokenizer = ScalaJniTokenizer("distilbert-base-cased")
    val tokenization = tokenizer.tokenize(words)

    val expectedTokens = Array("[CLS]", "EU", "rejects", "German", "call", "to", "boycott", "British", "la", "##mb", ".", "[SEP]")
    val actualTokens = tokenization.tokens
    actualTokens should contain theSameElementsInOrderAs expectedTokens

    val expectedTokenIds = Array(101, 7270, 22961, 1528, 1840, 1106, 21423, 1418, 2495, 12913, 119, 102)
    val actualTokenIds = tokenization.tokenIds
    actualTokenIds should contain theSameElementsInOrderAs expectedTokenIds

    val expectedWordIds = Array(-1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, -1)
    val actualWordIds = tokenization.wordIds
    actualWordIds should contain theSameElementsInOrderAs expectedWordIds
  }

  it should "tokenize with the bert-base-cased tokenizer" in {
    val tokenizer = ScalaJniTokenizer("bert-base-cased")
    val tokenization = tokenizer.tokenize(words)

    val expectedTokens = Array("[CLS]", "EU", "rejects", "German", "call", "to", "boycott", "British", "la", "##mb", ".", "[SEP]")
    val actualTokens = tokenization.tokens
    actualTokens should contain theSameElementsInOrderAs expectedTokens

    val expectedTokenIds = Array(101, 7270, 22961, 1528, 1840, 1106, 21423, 1418, 2495, 12913, 119, 102)
    val actualTokenIds = tokenization.tokenIds
    actualTokenIds should contain theSameElementsInOrderAs expectedTokenIds

    val expectedWordIds = Array(-1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, -1)
    val actualWordIds = tokenization.wordIds
    actualWordIds should contain theSameElementsInOrderAs expectedWordIds
  }

  it should "tokenize with the xlm-robert-base tokenizer" in {
    val tokenizer = ScalaJniTokenizer("xlm-roberta-base")
    val tokenization = tokenizer.tokenize(words)

    val expectedTokens = Array("<s>", "_EU", "_re", "ject", "s", "_German", "_call", "_to", "_boy", "cot", "t", "_British", "_la", "mb", "_", ".", "</s>")
        .map(word => word.replace('_', '\u2581'))
    val actualTokens = tokenization.tokens
    actualTokens should contain theSameElementsInOrderAs expectedTokens

    val expectedTokenIds = Array(0, 3747, 456, 75161, 7, 30839, 11782, 47, 25299, 47924, 18, 56101, 21, 6492, 6, 5, 2)
    val actualTokenIds = tokenization.tokenIds
    actualTokenIds should contain theSameElementsInOrderAs expectedTokenIds

    val expectedWordIds = Array(-1, 0, 1, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 7, 8, 8, -1)
    val actualWordIds = tokenization.wordIds
    actualWordIds should contain theSameElementsInOrderAs expectedWordIds
  }

  it should "tokenize with the roberta-base tokenizer with addPrefixSpace = true" in {
    val tokenizer = ScalaJniTokenizer("roberta-base", addPrefixSpace = true)
    val tokenization = tokenizer.tokenize(words)

    val expectedTokens = Array("<s>", "_EU", "_rejects", "_German", "_call", "_to", "_boycott", "_British", "_lamb", "_.", "</s>")
        .map(word => word.replace('_', '\u0120'))
    val actualTokens = tokenization.tokens
    actualTokens should contain theSameElementsInOrderAs expectedTokens

    val expectedTokenIds = Array(0, 1281, 24020, 1859, 486, 7, 13978, 1089, 17988, 479, 2)
    val actualTokenIds = tokenization.tokenIds
    actualTokenIds should contain theSameElementsInOrderAs expectedTokenIds

    val expectedWordIds = Array(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1)
    val actualWordIds = tokenization.wordIds
    actualWordIds should contain theSameElementsInOrderAs expectedWordIds
  }

  it should "tokenize with the roberta-base tokenizer with addPrefixSpace = false" in {
    val tokenizer = ScalaJniTokenizer("roberta-base", addPrefixSpace = false)
    val tokenization = tokenizer.tokenize(words)

    // The special symbols are not added and words are broken up.  I think this was added later.
    val expectedTokens = Array("<s>", "EU", "re", "ject", "s", "German", "call", "to", "boy", "cott", "British", "lam", "b", ".", "</s>")
    val actualTokens = tokenization.tokens
    actualTokens should contain theSameElementsInOrderAs expectedTokens

    val expectedTokenIds = Array(0, 14707, 241, 21517, 29, 27709, 16395, 560, 9902, 17083, 24270, 5112, 428, 4, 2)
    val actualTokenIds = tokenization.tokenIds
    actualTokenIds should contain theSameElementsInOrderAs expectedTokenIds

    val expectedWordIds = Array(-1, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8, -1)
    val actualWordIds = tokenization.wordIds
    actualWordIds should contain theSameElementsInOrderAs expectedWordIds
  }

  it should "tokenize with the microsoft/deberta-v3-base tokenizer" in {
    val tokenizer = ScalaJniTokenizer("./pretrained/microsoft-fast/tokenizer.json", addPrefixSpace = false)
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

  behavior of "LongTokenization"

  it should "have the same values as regular Tokenization" in {
    val tokenizer = ScalaJniTokenizer("xlm-roberta-base")
    val tokenization = tokenizer.tokenize(words)
    val longTokenization = LongTokenization(tokenization)

    longTokenization.tokenIds should contain theSameElementsInOrderAs tokenization.tokenIds
    longTokenization.wordIds should contain theSameElementsInOrderAs tokenization.wordIds
    longTokenization.tokens should contain theSameElementsInOrderAs tokenization.tokens
  }
}
