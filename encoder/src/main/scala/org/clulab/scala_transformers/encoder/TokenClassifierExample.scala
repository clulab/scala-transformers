package org.clulab.scala_transformers.encoder

import java.io.File

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.scala_transformers.tokenizer.LongTokenization

object TokenClassifierExample extends App {
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
  println("Predicted labels: " + labels.mkString(", "))
}
