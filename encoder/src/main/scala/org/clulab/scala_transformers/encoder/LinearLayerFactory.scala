package org.clulab.scala_transformers.encoder

import breeze.linalg.{DenseMatrix, DenseVector}
import org.slf4j.{Logger, LoggerFactory}

import scala.io.Source

abstract class LinearLayerFactory(val linearLayerLayout: LinearLayerLayout) extends Sourcer {
  lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def newSource(place: String): Source
  def exists(place: String): Boolean

  def name: String = sourceLine(newSource(linearLayerLayout.name))

  def dual: Boolean = sourceBoolean(newSource(linearLayerLayout.dual))

  def getWeights: DenseMatrix[Float] = {
    val place = linearLayerLayout.weights
    if (!exists(place))
      throw new RuntimeException(s"ERROR: you need at least a weights file for linear layer $name!")
    val values = sourceFloatMatrix(newSource(place))
    // dimensions: rows = hidden state size, columns = labels' count
    val weights = BreezeUtils.mkRowMatrix(values).t

    logger.info(s"Found weights with dimension ${weights.rows} x ${weights.cols}")
    weights
  }

  def getBiasesOpt: Option[DenseVector[Float]] = {
    val place = linearLayerLayout.biases
    if (exists(place)) {
      val values = sourceFloatVector(newSource(place))
      // the bias is a column vector
      val biases = DenseVector(values)

      logger.info(s"Found biases with dimension ${biases.length}")
      Some(biases)
    }
    else None
  }

  def getLabelsOpt: Option[Array[String]] = {
    val place = linearLayerLayout.labels
    if (exists(place)) {
      val values = sourceStringVector(newSource(place))

      Some(values)
    }
    else None
  }

  def newLinearLayer: LinearLayer = {
    val weights = getWeights
    val biasesOpt = getBiasesOpt
    val labelsOpt = getLabelsOpt

    new LinearLayer(name, dual, weights, biasesOpt, labelsOpt)
  }
}

class LinearLayerFactoryFromFiles(linearLayerLayout: LinearLayerLayout) extends LinearLayerFactory(linearLayerLayout) {

  def newSource(fileName: String): Source = Sourcer.sourceFromFile(fileName)

  def exists(fileName: String): Boolean = Sourcer.existsAsFile(fileName)
}

class LinearLayerFactoryFromResources(linearLayerLayout: LinearLayerLayout) extends LinearLayerFactory(linearLayerLayout) {

  def newSource(resourceName: String): Source = Sourcer.sourceFromResource(resourceName)

  // This should be a "file" rather than a "directory".
  def exists(resourceName: String): Boolean = Sourcer.existsAsResource(resourceName)
}