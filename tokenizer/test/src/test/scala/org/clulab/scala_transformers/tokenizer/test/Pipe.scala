package org.clulab.scala_transformers.tokenizer.test

abstract class Pipe {
  def write(text: String): Unit
  def read(): String
}
