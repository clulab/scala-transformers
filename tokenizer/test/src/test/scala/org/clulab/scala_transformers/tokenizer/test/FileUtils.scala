package org.clulab.scala_transformers.tokenizer.test

import java.io._
import java.nio.charset.StandardCharsets
import scala.io.{Codec, Source}
import scala.util.Using

class FileUtils(file: File, charsetName: String, append: Boolean = false) extends OutputStreamWriter(
  if (append) FileUtils.newAppendingBufferedOutputStream(file)
  else FileUtils.newBufferedOutputStream(file),
  charsetName
)

object FileUtils {
  val utf8: String = StandardCharsets.UTF_8.toString

  def printWriterFromFile(file: File, append: Boolean = false): PrintWriter =
    new PrintWriter(new FileUtils(file, utf8, append))

  def newAppendingBufferedOutputStream(file: File): BufferedOutputStream =
    new BufferedOutputStream(new FileOutputStream(file, true))

  def newBufferedOutputStream(file: File): BufferedOutputStream =
    new BufferedOutputStream(new FileOutputStream(file))

  def getTextFromResource(path: String): String =
    Using.resource(Source.fromResource(path)(Codec.UTF8)) { source =>
      source.mkString
    }

  def getTextFromFile(file: File): String =
    Using.resource(Source.fromFile(file)(Codec.UTF8)) { source =>
      source.mkString
    }
}
