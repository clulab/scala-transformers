package org.clulab.scala_transformers.tokenizer.jni.apps

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

// This is meant to exercise all tokenizers for which there is a resource.
// It verifies that the resource is in order.  They are stored under
// resources/org/clulab/scala_transformers/tokenizer.
object ScalaJniTokenizerApp extends App {
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val names = Array(
    "bert-base-cased",
    "distilbert-base-cased",
    "google/bert_uncased_L-4_H-512_A-8",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "roberta-base",
    "thomas-sounack/BioClinical-ModernBERT-base",
    "xlm-roberta-base"
  )

  println(s"words: ${words.mkString(" ")}")
  names.foreach { name =>
    val tokenizer = ScalaJniTokenizer(name)
    val tokenization = tokenizer.tokenize(words)

    println(tokenization)
  }
}
