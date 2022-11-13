package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.Tokenizer
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.scala_transformers.tokenizer.LongTokenization
import scala.collection.mutable.ArrayBuffer
import java.io.File

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
  val tokenizer: Tokenizer) {

  /** 
   * Predict labels for all tasks for a given sentence 
   * @param words Words in this sentence
   * @return Sequnce of labels for each task, for each token
   */
  def predict(words: Seq[String]): Seq[Seq[String]] = {
    // tokenize to subword tokens
    val tokenization = LongTokenization(tokenizer.tokenize(words.toArray))
    val inputIds = tokenization.tokenIds

    // run the sentence through the transformer encoder
    val encOutput = encoder.forward(inputIds)

    // now generate token label predictions for each task
    val allLabels = new ArrayBuffer[Seq[String]]
    for(task <- tasks) {
      val tokenLabels = task.predict(encOutput)
      val wordLabels = TokenClassifier.mapTokenLabelsToWords(tokenLabels, tokenization.wordIds)
      allLabels += wordLabels
    }

    return allLabels.toSeq
  }
}

object TokenClassifier {
  def apply(modelDir: String): TokenClassifier = {
    println(s"Loading TokenClassifier from directory $modelDir...")
    val encoder = Encoder(new File(s"$modelDir/encoder.onnx").getAbsolutePath().toString)
    val encName = readLine(new File(s"$modelDir/encoder.name"))
    val tokenizer = ScalaJniTokenizer(encName)

    val taskParentDir = new File(s"$modelDir/tasks")
    val taskDirs = taskParentDir.listFiles().map(_.getAbsolutePath).sorted
    val tasks = new ArrayBuffer[LinearLayer]()
    for(taskDir <- taskDirs) {
      println(s"Loading task from directory $taskDir...")
      tasks += LinearLayer(readLine(new File(s"$taskDir/name")), taskDir)
    }

    println("Load complete.")
    new TokenClassifier(encoder, tasks.toArray, tokenizer)
  }

  private def readLine(f: File): String = {
    val s = scala.io.Source.fromFile(f)
    val firstLine = s.getLines().next().trim()
    s.close()
    firstLine
  }

  private def mapTokenLabelsToWords(tokenLabels: Array[String], wordIds: Array[Long]): Seq[String] = {
    assert(tokenLabels.length == wordIds.length)
    val wordLabels = new ArrayBuffer[String]()
    var prevWordId = -1l
    for(i <- tokenLabels.indices) {
      val wordId = wordIds(i)
      if(wordId >= 0 && wordId != prevWordId) {
        wordLabels += tokenLabels(i)
      }
      prevWordId = wordId
    }
    wordLabels.toSeq
  }
}
