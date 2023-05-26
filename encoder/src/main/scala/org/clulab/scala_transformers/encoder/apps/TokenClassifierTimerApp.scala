package org.clulab.scala_transformers.encoder.apps

import org.clulab.scala_transformers.encoder.TokenClassifier
import org.clulab.scala_transformers.encoder.timer.Timers
import org.clulab.scala_transformers.tokenizer.LongTokenization

import java.io.File
import java.io.PrintWriter
import java.nio.charset.StandardCharsets
import scala.io.Source

class TokenClassifierTimer {
  lazy val tokenClassifier = new TimedTokenClassifier(TokenClassifier.fromFiles("../roberta-base-mtl/avg_export"))

  def readSentences(fileName: String): Seq[String] = {
    val source = Source.fromFile(fileName)

    try {
      source.getLines().toVector
    }
    finally {
      source.close()
    }
  }

  def readLabels(fileName: String): Seq[Seq[String]] = {
    val source = Source.fromFile(fileName)

    try {
      source.getLines().toVector.map { line =>
        line.split(' ').toSeq
      }
    }
    finally {
      source.close()
    }
  }

  def writeLabels(fileName: String, labelsCollection: Seq[Seq[String]]): Unit= {
    val printWriter = new PrintWriter(new File(fileName), StandardCharsets.UTF_8)

    try {
      labelsCollection.foreach { labels =>
        printWriter.println(labels.mkString(" "))
      }
    }
    finally {
      printWriter.close()
    }
  }

  def makeLabels(sentences: Seq[String]): Seq[Seq[String]] = {
    val elapsedTimer = Timers.getOrNew("Elapsed")
    val collectionOfLabels = elapsedTimer.time {
      sentences.zipWithIndex/*.par*/.map { case (sentence, index) =>
        println(s"$index $sentence")
        if (index != 1382) {
          val words = sentence.split(" ").toSeq

          // println(s"Words: ${words.mkString(" ")}")
          val labels = tokenClassifier.predict(words).flatten.toVector

          labels
        }
        else
          Vector.empty[String]
      }
    }

    Timers.summarize()
    collectionOfLabels.toVector
  }
}

class TimedTokenClassifier(tokenClassifier: TokenClassifier) extends TokenClassifier(
  tokenClassifier.encoder, tokenClassifier.tasks, tokenClassifier.tokenizer
) {
  val tokenizeTimer = Timers.getOrNew("Tokenizer")
  val forwardTimer = Timers.getOrNew("Encoder.forward")
  val predictTimers = tokenClassifier.tasks.indices.map { index =>
    Timers.getOrNew(s"Encoder.predict $index")
  }

  override def predict(words: Seq[String], headTaskName: String = "Deps Head"): Array[Array[String]] = {
    val headTaskIndexOpt = {
      val headTaskIndex = tasks.indexWhere { task => task.name == headTaskName && !task.dual }

      if (headTaskIndex >= 0) Some(headTaskIndex)
      else None
    }

    val tokenization = tokenizeTimer.time {
      LongTokenization(tokenizer.tokenize(words.toArray))
    }

    val encOutput = forwardTimer.time {
      val encOutput = encoder.forward(tokenization.tokenIds)

      encOutput
    }

    val notDualTokenAndWordLabels = tasks.zipWithIndex.map { case (task, index) =>
      if (!task.dual) {
        val tokenAndWordLabels = predictTimers(index).time {
          val tokenLabels = task.predict(encOutput, None, None)
          val wordLabels = TokenClassifier.mapTokenLabelsToWords(tokenLabels, tokenization.wordIds)

          (tokenLabels, wordLabels)
        }

        Some(tokenAndWordLabels)
      }
      else None
    }

    val dualTokenAndWordLabels = if (headTaskIndexOpt.isDefined) {
      val headsOpt = Some(notDualTokenAndWordLabels(headTaskIndexOpt.get).get._1.map { sth => sth.toInt })
      val masksOpt = Some(TokenClassifier.mkTokenMask(tokenization.wordIds))
      val dualTokenAndWordLabels = tasks.zipWithIndex.map { case (task, index) =>
        if (task.dual) {
          val tokenAndWordLabels = predictTimers(index).time {
            val tokenLabels = task.predict(encOutput, headsOpt, masksOpt)
            val wordLabels = TokenClassifier.mapTokenLabelsToWords(tokenLabels, tokenization.wordIds)

            (tokenLabels, wordLabels)
          }

          Some(tokenAndWordLabels)
        }
        else None
      }

      dualTokenAndWordLabels
    }
    else
      tasks.map { _ => None: Option[(Array[String], Array[String])] }

    val wordLabels = notDualTokenAndWordLabels.zip(dualTokenAndWordLabels).map { case (notDualTokenAndWordLabels, dualTokenAndWordLabels) =>
      if (notDualTokenAndWordLabels.isDefined) notDualTokenAndWordLabels.get._2
      else if (dualTokenAndWordLabels.isDefined) dualTokenAndWordLabels.get._2
      else throw new RuntimeException("Some task was unpredicted.")
    }

    wordLabels
  }
}

object TokenClassifierTimerApp extends App {
  val sentencesFileName = args.lift(0).getOrElse("../sentences.txt")
  val    labelsFileName = args.lift(1).getOrElse("../labels.txt")
  val tokenClassifierTimer = new TokenClassifierTimer()
  val sentences = tokenClassifierTimer.readSentences(sentencesFileName)
  val collectionOfLabels = tokenClassifierTimer.makeLabels(sentences)

  // Optionally write out the labels
  // tokenClassifierTimer.writeLabels(labelsFileName, collectionOfLabels)
}
