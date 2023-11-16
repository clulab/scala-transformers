package org.clulab.scala_transformers.apps

import org.clulab.scala_transformers.common.Timers
import org.clulab.scala_transformers.encoder.{EncoderMaxTokensRuntimeException, TokenClassifier}
import org.clulab.scala_transformers.tokenizer.LongTokenization

import scala.io.Source

object TokenClassifierTimerApp extends App {

  class TimedTokenClassifier(tokenClassifier: TokenClassifier) extends TokenClassifier(
    tokenClassifier.encoder, tokenClassifier.maxTokens, tokenClassifier.tasks, tokenClassifier.tokenizer
  ) {
    val tokenizeTimer = Timers.getOrNew("Tokenizer")
    val forwardTimer = Timers.getOrNew("Encoder.forward")
    val predictTimers = tokenClassifier.tasks.indices.map { index =>
      val name = tasks(index).name

      Timers.getOrNew(s"Encoder.predict $index\t$name")
    }

    // NOTE: This should be copied from the base class and then instrumented with timers.
    override def predictWithScores(words: Seq[String], headTaskName: String = "Deps Head"): Array[Array[Array[(String, Float)]]] = {
      // This condition must be met in order for allLabels to be filled properly without nulls.
      // The condition is not checked at runtime!
      // if (tasks.exists(_.dual))
      //   require(tasks.count(task => !task.dual && task.name == headTaskName) == 1)

      // tokenize to subword tokens
      val tokenization = tokenizeTimer.time {
        LongTokenization(tokenizer.tokenize(words.toArray))
      }
      val inputIds = tokenization.tokenIds
      val wordIds = tokenization.wordIds
      val tokens = tokenization.tokens

      if (inputIds.length > maxTokens) {
        throw new EncoderMaxTokensRuntimeException(s"Encoder error: the following text contains more tokens than the maximum number accepted by this encoder ($maxTokens): ${tokens.mkString(", ")}")
      }

      // run the sentence through the transformer encoder
      val encOutput = forwardTimer.time {
        encoder.forward(inputIds)
      }

      // outputs for all tasks stored here: task x tokens in sentence x scores per token
      val allLabels = new Array[Array[Array[(String, Float)]]](tasks.length)
      // all heads predicted for every token
      // dimensions: token x heads
      var heads: Option[Array[Array[Int]]] = None

      // now generate token label predictions for all primary tasks (not dual!)
      for (i <- tasks.indices) {
        if (!tasks(i).dual) {
          val tokenLabels = predictTimers(i).time {
            tasks(i).predictWithScores(encOutput, None, None)
          }
          val wordLabels = TokenClassifier.mapTokenLabelsAndScoresToWords(tokenLabels, tokenization.wordIds)
          allLabels(i) = wordLabels

          // if this is the task that predicts head positions, then save them for the dual tasks
          // we save all the heads predicted for each token
          if (tasks(i).name == headTaskName) {
            heads = Some(tokenLabels.map(_.map(_._1.toInt)))
          }
        }
      }

      // generate outputs for the dual tasks, if heads were predicted by one of the primary tasks
      // the dual task(s) must be aligned with the heads.
      //   that is, we predict the top label for each of the head candidates
      if (heads.isDefined) {
        //println("Tokens:    " + tokens.mkString(", "))
        //println("Heads:\n\t" + heads.get.map(_.slice(0, 3).mkString(", ")).mkString("\n\t"))
        //println("Masks:     " + TokenClassifier.mkTokenMask(wordIds).mkString(", "))
        val masks = Some(TokenClassifier.mkTokenMask(wordIds))

        for (i <- tasks.indices) {
          if (tasks(i).dual) {
            val tokenLabels = predictTimers(i).time {
              tasks(i).predictWithScores(encOutput, heads, masks)
            }
            val wordLabels = TokenClassifier.mapTokenLabelsAndScoresToWords(tokenLabels, tokenization.wordIds)
            allLabels(i) = wordLabels
          }
        }
      }

      allLabels
    }
  }

  val verbose = false
  val fileName = args.lift(0).getOrElse("../corpora/sentences/sentences.txt")
  val untimedTokenClassifier = TokenClassifier.fromFiles("../roberta-base-mtl-new/avg_export")
  val tokenClassifier = new TimedTokenClassifier(untimedTokenClassifier)
  val lines = {
    val source = Source.fromFile(fileName)
    val lines = source.getLines().take(100).toArray

    source.close
    lines
  }
  val elapsedTimer = Timers.getOrNew("Elapsed")

  elapsedTimer.time {
    lines.zipWithIndex/*.par*/.foreach { case (line, index) =>
      println(s"$index $line")
      if (index != 1382) {
        val words = line.split(" ").toSeq
        val allLabelSeqs = tokenClassifier.predictWithScores(words)

        if (verbose) {
          println(s"Words: ${words.mkString(", ")}")
          for (layer <- allLabelSeqs) {
            val words = layer.map(_.head) // Collapse the next layer by just taking the head.
            val wordLabels = words.map(_._1)

            println(s"Labels: ${wordLabels.mkString(", ")}")
          }
        }
      }
    }
  }
  Timers.summarize()
}
