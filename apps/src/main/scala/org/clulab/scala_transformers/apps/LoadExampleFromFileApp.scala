package org.clulab.scala_transformers.apps

import org.clulab.scala_transformers.encoder.{Encoder, TokenClassifierFactoryFromFiles, TokenClassifierLayout}
import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

object LoadExampleFromFileApp extends App {
  val baseName = args.lift(0).getOrElse("../tcmodel")
  val tokenClassifierLayout = new TokenClassifierLayout(baseName)
  val tokenClassifierFactory = new TokenClassifierFactoryFromFiles(tokenClassifierLayout)
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer(tokenClassifierFactory.name)
  val tokenization = tokenizer.tokenize(words)
  val inputIds = LongTokenization(tokenization).tokenIds
  val encoder = Encoder.fromFile(tokenClassifierLayout.onnx)
  val encOutput = encoder.forward(inputIds)

  println(encOutput)
}