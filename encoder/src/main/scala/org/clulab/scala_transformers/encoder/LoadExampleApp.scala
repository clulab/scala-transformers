package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

object LoadExampleApp extends App {
  val modelFilename = args.lift(0).getOrElse("../tcmodel/encoder.onnx")
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer("bert-base-cased")
  val tokenization = tokenizer.tokenize(words)
  val inputIds = LongTokenization(tokenization).tokenIds
  val encoder = Encoder(modelFilename)
  val encOutput = encoder.forward(inputIds)

  println(encOutput)
}