package org.clulab.scala_transformers.encoder.apps

import org.clulab.scala_transformers.encoder.TokenClassifier

/*
import java.io.File

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.scala_transformers.tokenizer.LongTokenization
*/

object TokenClassifierExampleApp extends App {
  //val tokenClassifier = TokenClassifier.fromFiles("../../scala-transformers-models/roberta-base-mtl/avg_export")
  val tokenClassifier = TokenClassifier.fromFiles("../microsoft-deberta-v3-base-mtl/avg_export")
  //val tokenClassifier = TokenClassifier.fromResources("/org/clulab/scala_transformers/models/microsoft_deberta_v3_base_mtl/avg_export")

  //val words = Seq("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val words = Seq("John", "Doe", "went", "to", "China", ".")
  println(s"Words: ${words.mkString(", ")}")

  println("The top label per token per task:")
  val allLabels = tokenClassifier.predict(words)
  for (labels <- allLabels) {
    if(labels != null) {
      println(s"Labels: ${labels.mkString(", ")}")
    }
  }

  println("Top 3 labels per token per task:")
  val allLabelsAndScores = tokenClassifier.predictWithScores(words)
  for (i <- allLabelsAndScores.indices) {
    println(s"Task #$i:")
    for (token <- allLabelsAndScores(i)) {
      println("\t" + token.slice(0, 3).mkString(", "))
    }
  }
}
