package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import scala.io.Source

abstract class TokenClassifierFactory(val tokenClassifierLayout: TokenClassifierLayout, val description: String) extends Sourcer {
  lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def newEncoder: Encoder
  def newLinearLayerFactory(index: Int): LinearLayerFactory

  def newSource(place: String): Source
  def exists(place: String): Boolean

  def name: String = readLine(newSource(tokenClassifierLayout.name))

  def taskCount: Int = 0.until(Int.MaxValue)
      .find { index =>
        val linearLayerLayout = tokenClassifierLayout.linearLayerLayout(index)
        val result = !exists(linearLayerLayout.name)

        result
      }
      .get

  def readLine(place: String): String = readLine(newSource(place))

  def readBoolean(place: String): Boolean = readBoolean(newSource(place))

  protected def newTokenClassifier(encoder: Encoder, tokenizerName: String, addPrefixSpace: Boolean, tasks: Array[LinearLayer]): TokenClassifier = {
    val tokenizer = ScalaJniTokenizer(tokenizerName, addPrefixSpace)

    new TokenClassifier(encoder, tasks, tokenizer)
  }

  def newTokenClassifier: TokenClassifier = {
    logger.info(s"Loading TokenClassifier from ${description} location ${tokenClassifierLayout.baseName}...")
    val addPrefixSpace = tokenClassifierLayout.addPrefixSpace
    val taskRange = 0.until(taskCount)
    val linearLayers = taskRange.map { taskIndex =>
      logger.info(s"Loading task from ${description} location ${tokenClassifierLayout.tasks}")
      val linearLayerFactory = newLinearLayerFactory(taskIndex)

      linearLayerFactory.newLinearLayer
    }
    val tokenClassifier = newTokenClassifier(newEncoder, name, addPrefixSpace, linearLayers.toArray)

    logger.info("Load complete.")
    tokenClassifier
  }
}

class TokenClassifierFactoryFromFiles(modelLayout: TokenClassifierLayout) extends TokenClassifierFactory(modelLayout, "filesystem") {

  def newSource(place: String): Source = Sourcer.sourceFromFile(place)

  def newEncoder: Encoder = Encoder.fromFile(new File(modelLayout.onnx).getAbsolutePath)

  def newLinearLayerFactory(index: Int) = new LinearLayerFactoryFromFiles(modelLayout.linearLayerLayout(index))

  def exists(place: String): Boolean = Sourcer.existsAsFile(place)
}

class TokenClassifierFactoryFromResources(modelLayout: TokenClassifierLayout) extends TokenClassifierFactory(modelLayout, "resource") {

  def newSource(place: String): Source = Sourcer.sourceFromResource(place)

  def newEncoder: Encoder = Encoder.fromResource(modelLayout.onnx)

  def newLinearLayerFactory(index: Int) = new LinearLayerFactoryFromResources(modelLayout.linearLayerLayout(index))

  // This should be a "file" rather than a "directory".
  def exists(place: String): Boolean = Sourcer.existsAsResource(place)
}
