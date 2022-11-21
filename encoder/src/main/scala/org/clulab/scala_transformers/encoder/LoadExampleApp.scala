package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer

import java.io.File

object LoadExampleApp extends App {
  val modelFilename = args.lift(0).getOrElse("../tcmodel/encoder.onnx")
  val tokenizerFilename = args.lift(1).getOrElse("../tcmodel/encoder.name")
  val tokenizerName = TokenClassifier.readLine(new File(tokenizerFilename))
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer(tokenizerName)
  val tokenization = tokenizer.tokenize(words)
  val inputIds = LongTokenization(tokenization).tokenIds
  val encoder = Encoder(modelFilename)
  val encOutput = encoder.forward(inputIds)

  println(encOutput)
}