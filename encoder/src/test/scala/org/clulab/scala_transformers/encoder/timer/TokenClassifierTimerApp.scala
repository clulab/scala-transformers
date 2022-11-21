package org.clulab.scala_transformers.encoder.timer

import org.clulab.scala_transformers.encoder.TokenClassifier
import org.clulab.scala_transformers.tokenizer.LongTokenization

import scala.io.Source

object TokenClassifierTimerApp extends App {
  val fileName = args.lift(0).getOrElse("../sentences.txt")

  class TimedTokenClassifier(tokenClassifier: TokenClassifier) extends TokenClassifier(
    tokenClassifier.encoder, tokenClassifier.tasks, tokenClassifier.tokenizer
  ) {
    val tokenizeTimer = Timers.getOrNew("Tokenizer")
    val forwardTimer = Timers.getOrNew("Encoder.forward")
    val predictTimers = tokenClassifier.tasks.indices.map { index =>
      Timers.getOrNew(s"Encoder.predict $index")
    }

    override def predict(words: Seq[String]): Array[Array[String]] = {

      val (tokenization, inputIds) = tokenizeTimer.time {
        val tokenization = LongTokenization(tokenizer.tokenize(words.toArray))
        val inputIds = tokenization.tokenIds

        (tokenization, inputIds)
      }

      val encOutput = forwardTimer.time {
        val encOutput = encoder.forward(inputIds)

        encOutput
      }

      val allLabels = tasks.zipWithIndex.map { case (task, index) =>
        val wordLabels = predictTimers(index).time {
          val tokenLabels = task.predict(encOutput)
          val wordLabels = TokenClassifier.mapTokenLabelsToWords(tokenLabels, tokenization.wordIds)

          wordLabels
        }

        wordLabels
      }

      allLabels
    }
  }

  val tokenClassifier = new TimedTokenClassifier(TokenClassifier("../tcmodel"))
  val lines = {
    val source = Source.fromFile(fileName)
    val lines = source.getLines.take(100).toArray

    source.close
    lines
  }
  val elapsedTimer = Timers.getOrNew("Elapsed")

  elapsedTimer.time {
    lines.zipWithIndex.foreach { case (line, index) =>
      println(s"$index $line")
      val words = line.split(" ")

      //    println(s"Words: ${words.mkString(", ")}")
      val allLabels = tokenClassifier.predict(words)
      //    for (labels <- allLabels)
      //      println(s"Labels: ${labels.mkString(", ")}")
    }
  }
  Timers.summarize
}
