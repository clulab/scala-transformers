package org.clulab.scala_transformers.tokenizer

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.transformers.test.Test

import scala.io.{Codec, Source}

class SentencesTest extends Test {
  // See also test_clu_tokenizer.py.
  val tokenizerNames = Seq(
    "bert-base-cased",
    "distilbert-base-cased",
    "roberta-base",
    "xlm-roberta-base",
    // All of these latter ones will not just fail, but cause a
    // fatal runtime error and end the testing completely.
    "google/bert_uncased_L-4_H-512_A-8",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-base"
  )

  behavior of "Tokenizer"

  def test(tokenizerName: String): Unit = {
    // Use this to get the tokenizer.
    val patchedTokenizerName =
        if (tokenizerName.contains("/")) "../pretrained/" + tokenizerName + "/tokenizer.json"
        else tokenizerName
    // Use this to get the sentence file.
    val modelName = tokenizerName.replace("/",  "-") + "-mtl"

    it should s"reproduce results for $tokenizerName" in {
      val addPrefixSpace = tokenizerName.contains("roberta")
      val tokenizer = ScalaJniTokenizer(patchedTokenizerName, addPrefixSpace)
      val inFileName = s"../corpora/sentences/$modelName.txt"
      val source = Source.fromFile(inFileName)(Codec.UTF8)

      try {
        source.getLines().grouped(3).foreach { lines =>
          val words = lines(0).split(" ")
          val tokenization = tokenizer.tokenize(words)
          val tokens = tokenization.tokens.map { rawToken =>
            val token = rawToken.replace("\\", "\\\\")

            if (token.contains("'")) s""""$token""""
            else s"'$token'"
          }
          val ids = tokenization.tokenIds
          val tokenString = tokens.mkString("[", ", ", "]")
          val idString = ids.mkString("[", ", ", "]")

          tokenString should be (lines(1))
          idString should be (lines(2))
        }
      }
      finally {
        source.close()
      }

      println(s"$tokenizerName has id ${tokenizer.tokenizerId}.")
    }
  }

  tokenizerNames.foreach(test)
}
