package org.clulab.scala_transformers.encoder

import java.io.File
import scala.io.{Codec, Source}

abstract class ModelReader(val modelLayout: ModelLayout, val description: String) {
  protected val codec = Codec.UTF8

  def newEncoder: Encoder
  def tokenizerName: String
  def taskName(index: Int): String
  def taskDual(index: Int): Boolean

  def exists(place: String): Boolean
  def taskCount: Int = 0.to(Int.MaxValue).find { index => !exists(modelLayout.getTaskName(index)) }.get - 1
}

class ModelReaderFromFiles(modelLayout: ModelLayout) extends ModelReader(modelLayout, "filesystem") {

  def newEncoder: Encoder = Encoder.fromFile(new File(modelLayout.getModel).getAbsolutePath)

  def tokenizerName = readLine(new File(modelLayout.getName))

  def taskName(index: Int): String = readLine(new File(modelLayout.getTaskName(index)))

  def taskDual(index: Int): Boolean = readBoolean(new File(modelLayout.getTaskDual(index)))

  def exists(place: String): Boolean = new File(place).exists

  def readLine(file: File): String = {
    val source = Source.fromFile(file)(codec)

    try {
      source.getLines.next.trim()
    }
    finally {
      source.close()
    }
  }

  def readBoolean(file: File): Boolean =
      readLine(file) == "1"
}

class ModelReaderFromResources(modelLayout: ModelLayout) extends ModelReader(modelLayout, "resource") {

  def newEncoder: Encoder = Encoder.fromResource(modelLayout.getModel)

  def tokenizerName = readLine(modelLayout.getName)

  def taskName(index: Int): String = readLine(modelLayout.getTaskName(index))

  def taskDual(index: Int): Boolean = readBoolean(modelLayout.getTaskDual(index))

  // This should be a "file" rather than a "directory".
  def exists(place: String): Boolean = {
    try {
      val connection = getClass.getResource(place).openConnection
      val contentLength = connection.getContentLength

      contentLength > 0
    }
    catch {
      case _: Throwable => false
    }
  }

  def readLine(resourceName: String): String = {
    val source = Source.fromResource(resourceName)(codec)

    try {
      source.getLines.next.trim()
    }
    finally {
      source.close()
    }
  }

  def readBoolean(resourceName: String): Boolean =
      readLine(resourceName) == "1"
}
