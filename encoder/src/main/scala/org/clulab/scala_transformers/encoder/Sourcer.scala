package org.clulab.scala_transformers.encoder

import java.io.File
import scala.io.{Codec, Source}

class Sourcer {

  protected def withSource[T](source: Source)(f: => T): T = {
    val result = try {
      f
    }
    finally {
      source.close()
    }

    result // debug here
  }

  def readLine(source: Source): String = withSource(source) {
    source.getLines.next.trim()
  }

  def readBoolean(source: Source): Boolean =
      readLine(source) == "1"

  def readFloatMatrix(source: Source): Array[Array[Float]] = withSource(source) {
    source
        .getLines
        .filterNot(_.startsWith("#"))
        .map(_.trim().split("\\s+").map(_.toFloat))
        .toArray
  }

  def readFloatVector(source: Source): Array[Float] = withSource(source) {
    source
        .getLines
        .filterNot(_.startsWith("#"))
        .flatMap(_.trim().split("\\s+"))
        .map(_.toFloat)
        .toArray
  }

  def readStringVector(source: Source): Array[String] = withSource(source) {
    source
        .getLines
        .map(_.trim)
        .toArray
  }
}

object Sourcer {
  protected val codec = Codec.UTF8

  def existsAsFile(fileName: String): Boolean = new File(fileName).exists

  def existsAsFile(file: File): Boolean = file.exists

  def existsAsResource(resourceName: String): Boolean = {
    try {
      val connection = getClass.getResource(resourceName).openConnection
      val contentLength = connection.getContentLength

      contentLength > 0
    }
    catch {
      case _: Throwable => false
    }
  }

  def sourceFromFile(file: File): Source = Source.fromFile(file)(codec)

  def sourceFromFile(fileName: String): Source = Source.fromFile(fileName)(codec)

  def sourceFromResource(resourceName: String): Source = Source.fromResource(resourceName)(codec)
}
