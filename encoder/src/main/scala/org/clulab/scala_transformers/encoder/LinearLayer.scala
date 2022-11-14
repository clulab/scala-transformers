package org.clulab.scala_transformers.encoder

import breeze.linalg.`*`
import breeze.linalg.argmax
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import scala.io.{Codec, Source}

/** Implements one linear layer */
class LinearLayer(
  val name: String,
  val weights: DenseMatrix[Float], // dimensions (hidden state size x labels size)
  val biasesOpt: Option[DenseVector[Float]], // column vector with length = labels size
  val labelsOpt: Option[Array[String]]
) {

  /** Forward pass for a single sentence */
  def forward(inputSentence: DenseMatrix[Float]): DenseMatrix[Float] = {
    val batch = Array(inputSentence)
    forward(batch).head
  }

  /**
   * Forward pass for a batch of sentences
   * @param inputBatch Each matrix in the batch has dimensions (sentence size x hidden state size)
   * @return Each output matrix has dimensions (sentence size x labels size)
   */
  def forward(inputBatch: Array[DenseMatrix[Float]]): Array[DenseMatrix[Float]] = {
    inputBatch.map { input =>
      //println("INPUT:\n" + input)
      val output = input * weights
      //println("OUTPUT before bias:\n" + output)
      for (b <- biasesOpt) output(*, ::) :+= b
      //println("OUTPUT after bias:\n" + output)
      output
    }
  }

  /** Predict the top label per token */
  def predict(inputSentence: DenseMatrix[Float]): Array[String] = {
    val batch = Array(inputSentence)
    predict(batch).head
  }

  /** Predict the top label for each token in each sentence in the batch */
  def predict(inputBatch: Array[DenseMatrix[Float]]): Array[Array[String]] = {
    val labels = labelsOpt.getOrElse(throw new RuntimeException("ERROR: can't predict without labels!"))
    // predict best label per (subword) token
    val logits = forward(inputBatch)
    val outputBatch = logits.map { logitsPerSentence =>
      // one token per row; pick argmax per token
      val bestLabels = Range(0, logitsPerSentence.rows).map { i =>
        val row = logitsPerSentence(i, ::)
        val bestIndex = argmax(row.t)

        labels(bestIndex)
      }

      bestLabels.toArray
    }

    outputBatch
  }
}

object LinearLayer {
  lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def apply(name: String, modelDir: String): LinearLayer = {
    val weightsFileName = modelDir + "/weights"
    val  biasesFileName = modelDir + "/biases"
    val  labelsFileName = modelDir + "/labels"

    if (!new File(weightsFileName).exists)
      throw new RuntimeException(s"ERROR: you need at least a weights file for linear layer $name!")
    val weights = loadWeights(weightsFileName)
    logger.info(s"Found weights with dimension ${weights.rows} x ${weights.cols}")

    val biasesOpt =
        if (new File(biasesFileName).exists) {
          val biases = loadBiases(biasesFileName)
          logger.info(s"Found biases with dimension ${biases.length}")
          Some(biases)
        }
        else None

    val labelsOpt =
        if (new File(labelsFileName).exists) {
          val labels = loadLabels(labelsFileName)
          logger.info(s"Using the following labels: ${labels.mkString(", ")}")
          Some(labels)
        }
        else None

    new LinearLayer(name, weights, biasesOpt, labelsOpt)
  }

  protected def loadWeights(fileName: String): DenseMatrix[Float] = {
    val source = Source.fromFile(fileName)(Codec.UTF8)

    try {
      val values = source
          .getLines
          .filterNot(_.startsWith("#"))
          .map(_.trim().split("\\s+").map(_.toFloat))
          .toArray

      // dimensions: rows = hidden state size, columns = labels' count
      BreezeUtils.mkRowMatrix(values).t
    }
    finally {
      source.close()
    }
  }

  protected def loadBiases(fileName: String): DenseVector[Float] = {
    val source = Source.fromFile(fileName)(Codec.UTF8)

    try {
      val values = source
          .getLines
          .filterNot(_.startsWith("#"))
          .flatMap(_.trim().split("\\s+"))
          .map(_.toFloat)
          .toArray

      // the bias is a column vector
      DenseVector(values)
    }
    finally {
      source.close()
    }
  }

  protected def loadLabels(fileName: String): Array[String] = {
    val source = Source.fromFile(fileName)(Codec.UTF8)

    try {
      source
          .getLines
          .map(_.trim)
          .toArray
    }
    finally {
      source.close()
    }
  }
}
