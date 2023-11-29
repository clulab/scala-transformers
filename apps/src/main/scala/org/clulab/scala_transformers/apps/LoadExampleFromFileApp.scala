package org.clulab.scala_transformers.apps

import org.clulab.scala_transformers.encoder.{Encoder, TokenClassifierFactoryFromFiles, TokenClassifierLayout}
import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

object LoadExampleFromFileApp extends App {
  // Choose one of these.
  val defaultBaseName = "../models/microsoft_deberta_v3_base_mtl/avg_export"
  // val defaultBaseName = "../models/google_electra_small_discriminator_mtl/avg_export"
  // val defaultBaseName = "../models/roberta_base_mtl/avg_export"

  val baseName = args.lift(0).getOrElse(defaultBaseName)
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