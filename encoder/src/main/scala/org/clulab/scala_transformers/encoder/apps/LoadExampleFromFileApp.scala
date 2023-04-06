package org.clulab.scala_transformers.encoder.apps

import org.clulab.scala_transformers.encoder.{Encoder, TokenClassifierLayout, TokenClassifierFactoryFromFiles, TokenClassifier}
import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

object LoadExampleFromFileApp extends App {
  val baseName = args.lift(0).getOrElse("../tcmodel")
  val modelLayout = new TokenClassifierLayout(baseName)
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer(modelLayout.name)
  val tokenization = tokenizer.tokenize(words)
  val inputIds = LongTokenization(tokenization).tokenIds
  val encoder = Encoder.fromFile(modelLayout.onnx)
  val encOutput = encoder.forward(inputIds)

  println(encOutput)
}