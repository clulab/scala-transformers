package org.clulab.scala_transformers.encoder

import java.io.File
import scala.io.{Codec, Source}

class Sourcer {

  protected def withSource[T](source: Source)(f: => T): T = {
    val result = try {
      f
    }
    finally {
      Option(source).foreach(_.close())
    }

    result // debug here
  }

  def sourceLine(source: Source): String = withSource(source) {
    source.getLines.next.trim()
  }

  def sourceBoolean(source: Source): Boolean =
      sourceLine(source) == "1"

  def sourceFloatMatrix(source: Source): Array[Array[Float]] = withSource(source) {
    source
        .getLines
        .toArray
        .filterNot(_.startsWith("#"))
        .map(_.trim().split("\\s+").map(_.toFloat))
  }

  def sourceFloatVector(source: Source): Array[Float] = withSource(source) {
    source
        .getLines
        .toArray
        .filterNot(_.startsWith("#"))
        .flatMap(_.trim().split("\\s+"))
        .map(_.toFloat)
  }

  def sourceStringVector(source: Source): Array[String] = withSource(source) {
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
      val resource = getClass.getResource(resourceName)
      val result = Option(resource).isDefined

      result
    }
    catch {
      case _: Throwable => false
    }
  }

  def sourceFromFile(file: File): Source = Source.fromFile(file)(codec)

  def sourceFromFile(fileName: String): Source = Source.fromFile(fileName)(codec)

  def sourceFromResource(resourceName: String): Source = {
    // For some reason, Source.fromResource() does not work here.
    val url = Sourcer.getClass.getResource(resourceName)
    val source = Source.fromURL(url)(codec)

    source
  }
}
