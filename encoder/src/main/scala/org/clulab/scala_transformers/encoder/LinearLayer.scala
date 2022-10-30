package org.clulab.scala_transformers.encoder

import breeze.linalg._
import java.io.File
import scala.collection.mutable.ArrayBuffer

/** Implements one linear layer */
class LinearLayer (name: String, 
  weights: DenseMatrix[Float], // dimensions (hidden state size x labels size)
  biases: Option[DenseVector[Float]], // column vector with length = labels size
  labels: Option[Array[String]]){

  /** Forward pass for a single sentence */
  def forward(inputSentence: DenseMatrix[Float]): DenseMatrix[Float] = {
    val batch = new Array[DenseMatrix[Float]](1)
    batch(0) = inputSentence
    forward(batch)(0)
  }

  /** 
   * Forward pass for a batch of sentences 
   * @param inputBatch Each matrix in the batch has dimensions (sentence size x hidden state size)
   * @return Each output matrix has dimensions (sentence size x labels size)
   */
  def forward(inputBatch: Array[DenseMatrix[Float]]): Array[DenseMatrix[Float]] = {
    val outputBatch = new ArrayBuffer[DenseMatrix[Float]]()
    for(input <- inputBatch) {
      val output = input * weights
      for(b <- biases) output(*, ::) :+= b
      outputBatch += output
    }
    outputBatch.toArray
  }  

  /** Predict the top label per token */
  def predict(inputSentence: DenseMatrix[Float]): Array[String] = {
    val batch = new Array[DenseMatrix[Float]](1)
    batch(0) = inputSentence
    predict(batch)(0)
  }

  /** Predict the top label for each token in each sentence in the batch */
  def predict(inputBatch: Array[DenseMatrix[Float]]): Array[Array[String]] = {
    if(labels.isEmpty) {
      throw new RuntimeException("ERROR: can't predict without labels!")
    }

    // predict best label per (subword) token
    val logits = forward(inputBatch)
    val outputBatch = new ArrayBuffer[Array[String]]()
    for(logitsPerSentence <- logits) {
      val predLabels = new ArrayBuffer[String]
      // one token per row; pick argmax per token
      for(i <- 0 until logitsPerSentence.rows) {
        val row = logitsPerSentence(i, ::)
        val bestIndex = argmax(row.t)
        val bestLabel = labels.get(bestIndex)
        predLabels += bestLabel
      }
      outputBatch += predLabels.toArray
    }

    outputBatch.toArray
  }
}

object LinearLayer {
  def apply(name: String, modelDir: String): LinearLayer = {
    val weightsFileName = modelDir + File.separator + "weights"
    val biasesFileName = modelDir + File.separator + "biases"
    val labelsFileName = modelDir + File.separator + "labels"

    if(! (new File(weightsFileName).exists())) {
      throw new RuntimeException(s"ERROR: you need at least a weights file for linear layer $name!")
    } 
    val weights = loadWeights(weightsFileName)
    println(s"Found weights with dimension ${weights.rows} x ${weights.cols}")

    val biases = if(new File(biasesFileName).exists()) 
      Some(loadBiases(biasesFileName)) else None
    if(biases.isDefined) {
      println(s"Found biases with dimension ${biases.get.length}")
    }

    val labels = if(new File(labelsFileName).exists())
      Some(loadLabels(labelsFileName)) else None
    if(labels.isDefined) {
      println(s"Using the following labels: ${labels.get.mkString(", ")}")
    }
    
    new LinearLayer(name, weights, biases, labels)
  }

  private def loadWeights(fn: String): DenseMatrix[Float] = {
    val source = io.Source.fromFile(fn)
    val values = new ArrayBuffer[Array[Float]]
    for(line <- source.getLines() if ! line.startsWith("#")) {
      val row = new ArrayBuffer[Float]
      val tokens = line.trim().split("\\s+")
      for(token <- tokens) {
        row += token.toFloat
      }
      values += row.toArray
    }
    source.close()
    // dimensions: rows = hidden state size, columns = labels' count
    BreezeUtils.mkRowMatrix(values.toArray).t
  }

  private def loadBiases(fn: String): DenseVector[Float] = {
    val source = io.Source.fromFile(fn)
    val values = new ArrayBuffer[Float]
    for(line <- source.getLines() if ! line.startsWith("#")) {
      val tokens = line.trim().split("\\s+")
      for(token <- tokens) {
        values += token.toFloat
      }
    }
    source.close()
    // the bias is a column vector
    DenseVector(values.toArray)
  }

  private def loadLabels(fn: String): Array[String] = {
    val source = io.Source.fromFile(fn)
    val labels = new ArrayBuffer[String]
    for(line <- source.getLines()) {
      labels += line.trim()
    }
    source.close()
    labels.toArray
  }
}
