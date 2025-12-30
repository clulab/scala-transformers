package org.clulab.scala_transformers.tokenizer

trait Tokenizing {
  def tokenize(words: Array[String]): Tokenization
}
