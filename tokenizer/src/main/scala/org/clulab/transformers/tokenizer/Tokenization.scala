package org.clulab.transformers.tokenizer

case class Tokenization(tokenIds: Array[Int], wordIds: Array[Int], tokens: Array[String]) {

  override def toString: String = {
    val tokenIdsString = tokenIds.mkString("[", ", ", "]")
    val wordIdsString = wordIds.mkString("[", ", ", "]")
    val tokensString = tokens.mkString("[\"", "\", \"", "\"]")

    s"tokenIds: $tokenIdsString\nwordIds: $wordIdsString\ntokens: $tokensString"
  }
}
