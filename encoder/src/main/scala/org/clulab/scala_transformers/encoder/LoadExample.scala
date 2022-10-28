package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

object LoadExample extends App {
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer("bert-base-cased")
  val tokenization = tokenizer.tokenize(words)

  // TODO: tokenizer should produce an Array[Long] not Int (ONNX expects Long)
  val inputIds = tokenization.tokenIds.map(x => x.toLong).toArray

  val enc = Encoder("/Users/msurdeanu/github/scala-transformers/encoder.onnx")
  val encOutput = enc.forward(inputIds)

  println(encOutput)
}