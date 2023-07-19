package org.clulab.scala_transformers.tokenizer

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.transformers.test.Test

class TokenizersTest extends Test {
  // See also test_clu_tokenizer.py.
  val tokenizerNames = Seq(
    "bert-base-cased",
    "distilbert-base-cased",
    "roberta-base",
    "xlm-roberta-base" // ,
    // All of these latter ones will not just fail, but cause a
    // fatal runtime error and end the testing completely.
    // "google/bert_uncased_L-4_H-512_A-8",
    // "google/electra-small-discriminator",
    // "microsoft/deberta-v3-base"
  )

  behavior of "Tokenizer"

  def test(tokenizerName: String): Unit = {
    it should s"$tokenizerName" in {
      val tokenizer = ScalaJniTokenizer(tokenizerName)

      println(s"$tokenizerName has id ${tokenizer.tokenizerId}.")
    }
  }

  tokenizerNames.foreach(test)
}
