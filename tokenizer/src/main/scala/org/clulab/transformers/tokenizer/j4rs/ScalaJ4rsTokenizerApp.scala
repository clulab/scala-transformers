package org.clulab.transformers.tokenizer.j4rs

object ScalaJ4rsTokenizerApp extends App {
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val names = Array(
    "distilbert-base-cased",
    "xlm-roberta-base"
  )

  println(s"words: ${words.mkString(" ")}")
  names.foreach { name =>
    val tokenizer = ScalaJ4rsTokenizer(name)
    val tokenization = tokenizer.tokenize(words)

    println(tokenization)
  }
}
