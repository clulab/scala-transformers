package org.clulab.scala_transformers.encoder.apps

import org.clulab.scala_transformers.encoder.TokenClassifier

/*
import java.io.File

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.scala_transformers.tokenizer.LongTokenization
*/

object TokenClassifierExampleApp extends App {
   val tokenClassifier = TokenClassifier.fromFiles("../roberta-base-mtl/avg_export")
//  val tokenClassifier = TokenClassifier.fromResources("/org/clulab/scala_transformers/models/roberta_base_mtl/avg_export")

  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  println(s"Words: ${words.mkString(", ")}")
  val allLabels = tokenClassifier.predict(words)
  for (labels <- allLabels) {
    if(labels != null) {
      println(s"Labels: ${labels.mkString(", ")}")
    }
  }

  /*
  val encoder = Encoder(new File("../encoder.onnx").getAbsolutePath().toString)
  val task = LinearLayer("NER", "..")

  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer("bert-base-cased")
  val tokenization = LongTokenization(tokenizer.tokenize(words))
  val inputIds = tokenization.tokenIds
  
  val encOutput = encoder.forward(inputIds)
  println(s"encOutput: ${encOutput.rows} x ${encOutput.cols}")

  val taskOutput = task.forward(encOutput)
  println(s"taskOutput: ${taskOutput.rows} x ${taskOutput.cols}")

  val labels = task.predict(encOutput)
  println("Tokens: " + tokenization.tokens.mkString(", "))
  println("Predicted labels: " + labels.mkString(", "))
  */
}
