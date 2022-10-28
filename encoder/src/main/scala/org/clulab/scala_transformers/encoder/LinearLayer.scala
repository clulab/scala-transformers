package org.clulab.scala_transformers.encoder

import breeze.linalg._
import java.io.File
import scala.collection.mutable.ArrayBuffer

/** Implements one linear layer */
class LinearLayer (name: String, weights: DenseMatrix[Float], biases: Option[DenseVector[Float]], labels: Option[Array[String]]){
  /** Forward pass for a single sentence */
  def forward(inputSentence: DenseMatrix[Float]): DenseMatrix[Float] = {
    val batch = new Array[DenseMatrix[Float]](1)
    batch(0) = inputSentence
    forward(batch)(0)
  }

  /** Forward pass for a batch of sentences */
  def forward(inputBatch: Array[DenseMatrix[Float]]): Array[DenseMatrix[Float]] = {
    null
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
    println("Found weights with dimension ${weights.rows} x ${weights.cols}")

    val biases = if(new File(biasesFileName).exists()) 
      Some(loadBiases(biasesFileName)) else None
    if(biases.isDefined) {
      println("Found biases with dimension ${biases.rows}")
    }

    val labels = if(new File(labelsFileName).exists())
      Some(loadLabels(labelsFileName)) else None
    
    new LinearLayer(name, weights, biases, labels)
  }

  def loadWeights(fn: String): DenseMatrix[Float] = {
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
    BreezeUtils.mkRowMatrix(values.toArray)
  }

  def loadBiases(fn: String): DenseVector[Float] = {
    val source = io.Source.fromFile(fn)
    val values = new ArrayBuffer[Float]
    for(line <- source.getLines() if ! line.startsWith("#")) {
      val tokens = line.trim().split("\\s+")
      for(token <- tokens) {
        values += token.toFloat
      }
    }
    source.close()
    DenseVector(values.toArray)
  }

  def loadLabels(fn: String): Array[String] = {
    val source = io.Source.fromFile(fn)
    val labels = new ArrayBuffer[String]
    for(line <- source.getLines()) {
      labels += line.trim()
    }
    source.close()
    labels.toArray
  }
}
