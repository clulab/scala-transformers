package org.clulab.scala_transformers.apps

import org.clulab.scala_transformers.encoder.{Encoder, TokenClassifierFactoryFromResources, TokenClassifierLayout}
import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

object LoadExampleFromResourceApp extends App {
  val baseName = "/org/clulab/scala_transformers/models/roberta_base_mtl/avg_export"
  val tokenClassifierLayout = new TokenClassifierLayout(baseName)
  val tokenClassifierFactory = new TokenClassifierFactoryFromResources(tokenClassifierLayout)
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer(tokenClassifierFactory.name)
  val tokenization = tokenizer.tokenize(words)
  val inputIds = LongTokenization(tokenization).tokenIds
  val encoder = Encoder.fromResource(tokenClassifierLayout.onnx)
  val encOutput = encoder.forward(inputIds)

  println(encOutput)
}