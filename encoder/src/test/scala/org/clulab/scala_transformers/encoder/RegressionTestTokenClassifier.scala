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
  val sentencesFileName = "./src/test/resources/sentences.txt"
  val labelsFileName = "./src/test/resources/labels.txt"
  val vectorsFileName = "./src/test/resources/vectors.txt"
  val sentences = tokenClassifierTimer.readSentences(sentencesFileName)

  // The model for the token classifier is not usually available, so this test is ignored by default.
  it should "produce consistent results" in {
    val expectedCollectionOfLabels = tokenClassifierTimer.readLabels(labelsFileName)
    val actualCollectionOfLabels = tokenClassifierTimer.makeLabels(sentences)
    val failingIndexes = actualCollectionOfLabels.zip(expectedCollectionOfLabels).zipWithIndex.filter { case ((actualLabels, expectedLabels), index) =>
      val failing = actualLabels != expectedLabels

      if (failing) {
        val message = s"""$index
            |  actual: ${actualLabels.mkString(" ")}
            |expected: ${expectedLabels.mkString(" ")}
            |""".stripMargin

        println(message)
      }
      failing
    }

    failingIndexes should be (empty)
  }

  it should "produce consistent vectors" in {
    val expectedCollectionOfVectors = tokenClassifierTimer.readVectors(vectorsFileName)
    val actualCollectionOfVectors = tokenClassifierTimer.makeVectors(sentences.take(expectedCollectionOfVectors.length))
    val failingIndexes = actualCollectionOfVectors.zip(expectedCollectionOfVectors).zipWithIndex.filter { case ((actualVectors, expectedVectors), index) =>
      val failing = actualVectors != expectedVectors

      if (failing) {
        val message =
          s"""$index
             |  actual: ${actualVectors.mkString(" ")}
             |expected: ${expectedVectors.mkString(" ")}
             |""".stripMargin

        println(message)
      }
      failing
    }

    failingIndexes should be(empty)
  }
}
