package org.clulab.scala_transformers.encoder.apps

import org.clulab.scala_transformers.encoder.{Encoder, TokenClassifier}
import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

import java.io.File

object LoadExampleFromResourceApp extends App {
  val modelResourceName = args.lift(0).getOrElse("/org/clulab/scala_transformers/models/roberta_base_mtl/avg_export/encoder.onnx")
  val tokenizerFilename = args.lift(1).getOrElse("../tcmodel/encoder.name")
  val tokenizerName = TokenClassifier.readLine(new File(tokenizerFilename))
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer(tokenizerName)
  val tokenization = tokenizer.tokenize(words)
  val inputIds = LongTokenization(tokenization).tokenIds
  val encoder = Encoder.fromResource(modelResourceName)
  val encOutput = encoder.forward(inputIds)

  println(encOutput)
}