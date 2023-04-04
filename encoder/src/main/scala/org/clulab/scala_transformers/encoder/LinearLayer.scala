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
  val dual: Boolean, 
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
  def predict(inputSentence: DenseMatrix[Float], 
              heads: Option[Array[Int]], 
              masks: Option[Array[Boolean]]): Array[String] = {
    val batchSentences = Array(inputSentence)
    val batchHeads = if(heads.isDefined) Some(Array(heads.get)) else None
    val batchMasks = if(masks.isDefined) Some(Array(masks.get)) else None
    predict(batchSentences, batchHeads, batchMasks).head
  }

  /** Predict the top label for each token in each sentence in the batch */
  def predict(inputBatch: Array[DenseMatrix[Float]], 
              batchHeads: Option[Array[Array[Int]]],
              batchMasks: Option[Array[Array[Boolean]]]): Array[Array[String]] = {
    if(dual) predictDual(inputBatch, batchHeads, batchMasks)
    else predictPrimal(inputBatch)
  }

  def concatenateModifiersAndHeads(
    sentenceHiddenStates: DenseMatrix[Float], 
    headRelativePositions: Array[Int]): DenseMatrix[Float] = {

    // this matrix concatenates the hidden states of modifier + corresponding head
    // rows = number of tokens in the sentence; cols = hidden state size
    val concatMatrix = DenseMatrix.zeros[Float](rows = sentenceHiddenStates.rows, cols = 2 * sentenceHiddenStates.cols)

    // traverse all modifiers
    for(i <- 0 until sentenceHiddenStates.rows) {
      val modHiddenState = sentenceHiddenStates(i, ::)
      // what is the absolute position of the head token in the sentence?
      val rawHeadAbsPos = i + headRelativePositions(i)
      val headAbsolutePosition = 
        if(rawHeadAbsPos >= 0 && rawHeadAbsPos < sentenceHiddenStates.rows) rawHeadAbsPos
        else i // if the absolute position is invalid (e.g., root node or incorrect prediction) duplicate the mod embedding
      val headHiddenState = sentenceHiddenStates(headAbsolutePosition, ::)

      // vector concatenation in Breeze operates over vertical vectors, hence the transposing here
      val concatState = DenseVector.vertcat(modHiddenState.t, headHiddenState.t).t

      // row i in the concatenated matrix contains the embedding of modifier i and its head
      concatMatrix(i, ::) :+= concatState
    }
    
    concatMatrix
  }

  /** Predict the top label for each combination of modifier token and corresponding head token */
  def predictDual(inputBatch: Array[DenseMatrix[Float]], 
                  batchHeads: Option[Array[Array[Int]]] = None,
                  batchMasks: Option[Array[Array[Boolean]]] = None): Array[Array[String]] = {
    assert(batchHeads.isDefined)
    assert(batchMasks.isDefined)
    val indexToLabel = labelsOpt.getOrElse(throw new RuntimeException("ERROR: can't predict without labels!"))

    val outputBatch = new Array[Array[String]](inputBatch.length)

    // we process one sentence at a time because the dual setting makes it harder to batch
    for(i <- inputBatch.indices) {
      val input = inputBatch(i)
      val heads = batchHeads.get(i)
      val masks = batchMasks.get(i)

      // generate a matrix that is twice as wide to concatenate the embeddings of the mod + head
      val concatInput = concatenateModifiersAndHeads(input, heads)

      // get the logits for the current sentence produced by this linear layer
      val logitsPerSentence = forward(Array(concatInput))(0)

      // one token per row; pick argmax per token
      val bestLabels = Range(0, logitsPerSentence.rows).map { i =>
        val row = logitsPerSentence(i, ::) // picks line i from a 2D matrix
        val bestIndex = argmax(row.t)
        indexToLabel(bestIndex)
      }

      outputBatch(i) = bestLabels.toArray
    }

    outputBatch                    
  }

  def predictPrimal(inputBatch: Array[DenseMatrix[Float]]): Array[Array[String]] = {
    val labels = labelsOpt.getOrElse(throw new RuntimeException("ERROR: can't predict without labels!"))
    // predict best label per (subword) token
    val logits = forward(inputBatch)
    val outputBatch = logits.map { logitsPerSentence =>
      // one token per row; pick argmax per token
      val bestLabels = Range(0, logitsPerSentence.rows).map { i =>
        val row = logitsPerSentence(i, ::) // picks line i from a 2D matrix
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

  def apply(name: String, dual: Boolean, modelDir: String): LinearLayer = {
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

    new LinearLayer(name, dual, weights, biasesOpt, labelsOpt)
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
