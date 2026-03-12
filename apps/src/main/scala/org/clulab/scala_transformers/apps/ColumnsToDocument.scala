package org.clulab.scala_transformers.apps

import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

private class ColumnsToDocument

class Document (val sentences: Seq[Sentence])

class Sentence (val words: Seq[String], val labels: Seq[String])

/**
  * Converts the CoNLLX column-based format to Document
  */
object ColumnsToDocument {
  val logger: Logger = LoggerFactory.getLogger(classOf[ColumnsToDocument])

  def readFromFile(
    fn: String,
    wordPos: Int,
    labelPos: Int
  ): Document = {
    val source = Source.fromFile(fn)
    readFromSource(source, wordPos, labelPos)
  }

  def readFromSource(
    source: Source,
    wordPos: Int,
    labelPos: Int
  ): Document = {
    val words = new ArrayBuffer[String]()
    val labels = new ArrayBuffer[String]()
    val sentences = new ArrayBuffer[Sentence]()

    def mkSentence(): Sentence = {
      val sent = new Sentence(words.toArray.toSeq, labels.toArray.toSeq)
      words.clear()
      labels.clear()
      sent
    }

    source.getLines().map(_.trim).foreach { l =>
      if (l.isEmpty) {
        // end of sentence
        if (words.nonEmpty) {
          sentences += mkSentence()
        }
      }
      else {
        // within the same sentence
        val bits = l.split("\\s+")
        if (bits.length < 2)
          throw new RuntimeException(s"ERROR: invalid line [$l]!")

        words += bits(wordPos)
        labels += bits(labelPos)
      }
    }
    if (words.nonEmpty) {
      val sent = mkSentence()
      sentences += sent
      //println("words: " + sent.words.mkString(", "))
      //println("labels: " + sent.entities.get.mkString(", "))
    }
    logger.debug(s"Loaded ${sentences.size} sentences.")

    new Document(sentences)
  }
}
