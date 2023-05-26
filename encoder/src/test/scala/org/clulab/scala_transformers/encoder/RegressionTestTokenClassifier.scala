package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.encoder.apps.TokenClassifierTimer
import org.clulab.transformers.test.Test

class RegressionTestTokenClassifier extends Test {
  behavior of "TokenClassifier"

  val tokenClassifierTimer = new TokenClassifierTimer()
  val sentencesFileName = "../sentences.txt"
  val labelsFileName = "../labels.txt"
  val sentences = tokenClassifierTimer.readSentences(sentencesFileName)
  val expectedCollectionOfLabels = tokenClassifierTimer.readLabels(labelsFileName)

  it should "produce consistent results" in {
    val actualCollectionOfLabels = tokenClassifierTimer.makeLabels(sentences)

    actualCollectionOfLabels.zip(expectedCollectionOfLabels).zipWithIndex.foreach { case ((actualLabels, expectedLabels), index) =>
      (index, expectedLabels.mkString(" ")) should be ((index, actualLabels.mkString(" ")))
    }
  }
}
