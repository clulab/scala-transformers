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

  def name: String = sourceLine(newSource(tokenClassifierLayout.name))
  def maxTokens: Int = try{
    sourceLine(newSource(tokenClassifierLayout.maxTokens)).toInt
  } catch {
    case e: Exception => 
      logger.error(s"Could not find the ${tokenClassifierLayout.maxTokens} file. Will assume a default value of ${TokenClassifierFactory.DEFAULT_MAX_TOKENS} here.")
      TokenClassifierFactory.DEFAULT_MAX_TOKENS
  }

  def taskCount: Int = 0.until(Int.MaxValue)
      .find { index =>
        val linearLayerLayout = tokenClassifierLayout.linearLayerLayout(index)
        val result = !exists(linearLayerLayout.name)

        result
      }
      .get

  def sourceLine(place: String): String = sourceLine(newSource(place))

  def sourceBoolean(place: String): Boolean = sourceBoolean(newSource(place))

  protected def newTokenClassifier(
    encoder: Encoder, 
    tokenizerName: String, 
    encoderMaxTokens: Int, 
    addPrefixSpace: Boolean, 
    tasks: Array[LinearLayer]): TokenClassifier = {

    val tokenizer = ScalaJniTokenizer(tokenizerName, addPrefixSpace)

    new TokenClassifier(encoder, encoderMaxTokens, tasks, tokenizer)
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
    val tokenClassifier = newTokenClassifier(newEncoder, name, maxTokens, addPrefixSpace, linearLayers.toArray)

    logger.info("Load complete.")
    tokenClassifier
  }
}

object TokenClassifierFactory {
  // most encoders used this as the max number of tokens
  val DEFAULT_MAX_TOKENS = 512
}

class TokenClassifierFactoryFromFiles(modelLayout: TokenClassifierLayout) extends TokenClassifierFactory(modelLayout, "filesystem") {

  def newSource(fileName: String): Source = Sourcer.sourceFromFile(fileName)

  def newEncoder: Encoder = Encoder.fromFile(new File(modelLayout.onnx).getAbsolutePath)

  def newLinearLayerFactory(index: Int) = new LinearLayerFactoryFromFiles(modelLayout.linearLayerLayout(index))

  def exists(fileName: String): Boolean = Sourcer.existsAsFile(fileName)
}

class TokenClassifierFactoryFromResources(modelLayout: TokenClassifierLayout) extends TokenClassifierFactory(modelLayout, "resource") {

  def newSource(resourceName: String): Source = Sourcer.sourceFromResource(resourceName)

  def newEncoder: Encoder = Encoder.fromResource(modelLayout.onnx)

  def newLinearLayerFactory(index: Int) = new LinearLayerFactoryFromResources(modelLayout.linearLayerLayout(index))

  // This should be a "file" rather than a "directory".
  def exists(resourceName: String): Boolean = Sourcer.existsAsResource(resourceName)
}
