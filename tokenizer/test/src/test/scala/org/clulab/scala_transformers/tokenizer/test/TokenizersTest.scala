package org.clulab.scala_transformers.tokenizer.test

import org.clulab.scala_transformers.tokenizer.scala.tokenizers.ScalaTokenizer
import org.clulab.transformers.test.Test

class TokenizersTest extends Test {
  val tokenizerNames = Seq(
    // This is a local file.  Use ../ for sbt.
    "../tokenizer/src/main/resources/org/clulab/scala_transformers/tokenizer/bert-base-cased/tokenizer.json",
    // These are all resources.
    // See also names.py.
    "microsoft/deberta-v3-base"
  )

  behavior of "Tokenizer"

  def test(tokenizerName: String): Unit = {
    it should s"""created a working "$tokenizerName" tokenizer""" in {
      val addPrefixSpace = tokenizerName.contains("roberta")
      val tokenizer = ScalaTokenizer(tokenizerName, addPrefixSpace)

//      println(s"$tokenizerName has id ${tokenizer.tokenizerId}.")

      val tokenization = tokenizer.tokenize(Array("This", "is", "a", "test", "."))
      println(tokenization)
    }
  }

  it should "not create an non-existent tokenizer" in {
    assertThrows[RuntimeException] {
      ScalaTokenizer("nonexistent")
    }
  }

  tokenizerNames.foreach(test)
}
