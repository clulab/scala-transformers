package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.Tokenizer
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import scala.io.{Codec, Source}

/** 
 * Implements the inference step of a token classifier for multi-task learning
 * The classifier uses a single encoder to generate the hidden state representation for every token and
 *   one linear classifier per task to produce task-specific token labels. Note that the encoder is 
 * The token classifier expects the model to be loaded to saved in one directory in the following format:
 * encoder.onnx - ONNX-formatted encoder model
 * encoder.name - Hugging Face name of the transformer used as the encoder
 * tasks/
 *   0/
 *     name - task name 
 *     weights - weights of the linear classifier
 *     biases - biases of the linear classifier
 *     labels - array of labels to be predicted
 *   1/
 *     ... (same as task 0)
 *   2/
 *     ... (same as task 0)
 *   ...
 */
class TokenClassifier(
  val encoder: Encoder, 
  val tasks: Array[LinearLayer],
  val tokenizer: Tokenizer
) {

  /** 
   * Predict labels for all tasks for a given sentence 
   * @param words Words in this sentence
   * @return Sequnce of labels for each task, for each token
   */
  def predict(words: Seq[String]): Array[Array[String]] = {
    // tokenize to subword tokens
    val tokenization = LongTokenization(tokenizer.tokenize(words.toArray))
    val inputIds = tokenization.tokenIds

    // run the sentence through the transformer encoder
    val encOutput = encoder.forward(inputIds)

    // now generate token label predictions for each task
    val allLabels = tasks.map { task =>
      if(task.dual == false) {
        val tokenLabels = task.predict(encOutput)
        val wordLabels = TokenClassifier.mapTokenLabelsToWords(tokenLabels, tokenization.wordIds)
        wordLabels
      } else {
        null // TODO!
      }
    }

    allLabels
  }
}

object TokenClassifier {
  lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def apply(modelDir: String): TokenClassifier = {
    logger.info(s"Loading TokenClassifier from directory $modelDir...")
    val encoder = Encoder(new File(s"$modelDir/encoder.onnx").getAbsolutePath())
    val tokenizerName = readLine(new File(s"$modelDir/encoder.name"))
    val tokenizer = ScalaJniTokenizer(tokenizerName)

    val taskParentDir = new File(s"$modelDir/tasks")
    val taskDirs = taskParentDir.listFiles().map(_.getAbsolutePath).sorted
    val tasks = taskDirs.map { taskDir =>
      logger.info(s"Loading task from directory $taskDir...")
      LinearLayer(readLine(new File(s"$taskDir/name")), readBoolean(new File(s"$taskDir/dual")), taskDir)
    }

    logger.info("Load complete.")
    new TokenClassifier(encoder, tasks, tokenizer)
  }

  def readLine(file: File): String = {
    val source = Source.fromFile(file)(Codec.UTF8)

    try {
      source.getLines.next.trim()
    }
    finally {
      source.close()
    }
  }

  def readBoolean(file: File): Boolean = {
    val s = readLine(file)
    if (s == "1") true else false
  }

  def mapTokenLabelsToWords(tokenLabels: Array[String], wordIds: Array[Long]): Array[String] = {
    require(tokenLabels.length == wordIds.length)
    val wordLabelOpts = tokenLabels.zip(wordIds).zipWithIndex.map { case ((tokenLabel, wordId), index) =>
      val valid = wordId >= 0 && (index == 0 || wordId != wordIds(index - 1))

      if (valid) Some(tokenLabel)
      else None
    }

    wordLabelOpts.flatten
  }
}
