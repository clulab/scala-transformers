package org.clulab.scala_transformers.tokenizer

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.transformers.test.Test

class TokenizersTest extends Test {
  val tokenizerNames = Seq(
    // This is a local file.  Use ../ for sbt.
    "../tokenizer/src/main/resources/org/clulab/scala_transformers/tokenizer/bert-base-cased/tokenizer.json",
    // These are all resources.
    // See also names.py.
    "bert-base-cased",
    "distilbert-base-cased",
    "roberta-base",
    "xlm-roberta-base",
    "google/bert_uncased_L-4_H-512_A-8",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-base",
    // Here are some that will be on the network.
    "roberta-large-mnli"
  )

  behavior of "Tokenizer"

  def test(tokenizerName: String): Unit = {
    it should s"""created a working "$tokenizerName" tokenizer""" in {
      val addPrefixSpace = tokenizerName.contains("roberta")
      val tokenizer = ScalaJniTokenizer(tokenizerName, addPrefixSpace)

      println(s"$tokenizerName has id ${tokenizer.tokenizerId}.")

      val tokenization = tokenizer.tokenize(Array("This", "is", "a", "test", "."))
      println(tokenization)
    }
  }

  it should "not create an non-existent tokenizer" in {
    assertThrows[RuntimeException] {
      ScalaJniTokenizer("nonexistent")
    }
  }

  tokenizerNames.foreach(test)
}
