package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.encoder.apps.TokenClassifierTimer
import org.clulab.transformers.test.Test

import java.io.File

class RegressionTestTokenClassifier extends Test {
  behavior of "TokenClassifier"

  // sbt by default uses the subproject directory, but IntelliJ needs to be configured
  // for that by using as the working directory $ContentRoot$.
  println(new File(".").getAbsolutePath)

  val tokenClassifierTimer = new TokenClassifierTimer()
  val sentencesFileName = "../encoder/src/test/resources/sentences.txt"
  val labelsFileName = "../encoder/src/test/resources/labels.txt"
  val sentences = tokenClassifierTimer.readSentences(sentencesFileName)
  val expectedCollectionOfLabels = tokenClassifierTimer.readLabels(labelsFileName)

  // The model for the token classifier is not usually available, so this test is ignored by default.
  ignore should "produce consistent results" in {
    val actualCollectionOfLabels = tokenClassifierTimer.makeLabels(sentences)

    actualCollectionOfLabels.zip(expectedCollectionOfLabels).zipWithIndex.foreach { case ((actualLabels, expectedLabels), index) =>
      (index, expectedLabels.mkString(" ")) should be ((index, actualLabels.mkString(" ")))
    }
  }
}
