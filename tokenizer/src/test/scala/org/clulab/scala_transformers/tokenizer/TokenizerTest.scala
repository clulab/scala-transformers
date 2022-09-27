package org.clulab.scala_transformers.tokenizer

import org.clulab.scala_transformers.tokenizer.j4rs.ScalaJ4rsTokenizer
import org.clulab.transformers.test.Test

class TokenizerTest extends Test {
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")

  behavior of "Tokenizer"

  it should "tokenize with the distilbert-base-cased tokenizer" in {
    val expectedTokens = Array("[CLS]", "EU", "rejects", "German", "call", "to", "boycott", "British", "la", "##mb", ".", "[SEP]")
    val tokenizer = ScalaJ4rsTokenizer("distilbert-base-cased")
    val tokenization = tokenizer.tokenize(words)
    val actualTokens = tokenization.tokens

    actualTokens should contain theSameElementsInOrderAs expectedTokens
  }

  it should "tokenize with the xlm-robert-base tokenizer" in {
    val expectedTokens = Array("<s>", "_EU", "_re", "ject", "s", "_German", "_call", "_to", "_boy", "cot", "t", "_British", "_la", "mb", "_", ".", "</s>")
        .map(word => word.replace('_', '\u2581'))
    val tokenizer = ScalaJ4rsTokenizer("xlm-roberta-base")
    val tokenization = tokenizer.tokenize(words)
    val actualTokens = tokenization.tokens

    actualTokens should contain theSameElementsInOrderAs expectedTokens
  }
}
