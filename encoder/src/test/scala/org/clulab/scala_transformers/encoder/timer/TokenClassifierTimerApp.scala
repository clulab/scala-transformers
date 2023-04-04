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

  val tokenClassifier = new TimedTokenClassifier(TokenClassifier("../tcmodel"))
  val lines = {
    val source = Source.fromFile(fileName)
    val lines = source.getLines.toArray

    source.close
    lines
  }
  val elapsedTimer = Timers.getOrNew("Elapsed")

  elapsedTimer.time {
    lines.zipWithIndex/*.par*/.foreach { case (line, index) =>
      println(s"$index $line")
      if (index != 1382) {
        val words = line.split(" ")

        //    println(s"Words: ${words.mkString(", ")}")
        val allLabels = tokenClassifier.predict(words)
        //    for (labels <- allLabels)
        //      println(s"Labels: ${labels.mkString(", ")}")
      }
    }
  }
  Timers.summarize
}
