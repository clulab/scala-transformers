package org.clulab.scala_transformers.tokenizer

case class LongTokenization(tokenIds: Array[Long], wordIds: Array[Long], tokens: Array[String]) {

  override def toString: String = {
    val tokenIdsString = tokenIds.mkString("[", ", ", "]")
    val wordIdsString = wordIds.mkString("[", ", ", "]")
    val tokensString = tokens.mkString("[\"", "\", \"", "\"]")

    s"tokenIds: $tokenIdsString\nwordIds: $wordIdsString\ntokens: $tokensString"
  }
}

object LongTokenization {

  def apply(tokenization: Tokenization): LongTokenization = LongTokenization(
    tokenization.tokenIds.map(_.toLong),
    tokenization.wordIds.map(_.toLong),
    tokenization.tokens
  )
}
