package org.clulab.transformers.tokenizer

trait Tokenizing {
  def tokenize(words: Array[String]): Tokenization
}
