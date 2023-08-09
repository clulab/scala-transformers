package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.Tokenizer
import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.scala_transformers.tokenizer.LongTokenization
import org.slf4j.{Logger, LoggerFactory}

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
    * Predict labels together with their scores for all tasks for a given sentence 
    *
    * @param words Words in this sentence
    * @param headTaskName Which tasks indicates the predictions for dependency heads (if any)
    * @return Labels and scores. Dimensions are: tasks x tokens in the sentence x array of (label, logit) per token
    */
  def predictWithScores(words: Seq[String], headTaskName:String = "Deps Head"): Array[Array[Array[(String, Float)]]] = {
    // tokenize to subword tokens
    val tokenization = LongTokenization(tokenizer.tokenize(words.toArray))
    val inputIds = tokenization.tokenIds
    val wordIds = tokenization.wordIds
    val tokens = tokenization.tokens

    // run the sentence through the transformer encoder
    val encOutput = encoder.forward(inputIds)

    // outputs for all tasks stored here: task x tokens in sentence x scores per token
    val allLabels = new Array[Array[Array[(String, Float)]]](tasks.length)
    var heads: Option[Array[Int]] = None

    // now generate token label predictions for all primary tasks (not dual!)
    for(i <- tasks.indices) {
      if(! tasks(i).dual) {
        val tokenLabels = tasks(i).predictWithScores(encOutput, None, None)
        val wordLabels = TokenClassifier.mapTokenLabelsAndScoresToWords(tokenLabels, tokenization.wordIds)
        allLabels(i) = wordLabels

        // if this is the task that predicts head positions, then save them for the dual tasks
        // here we save only the head predicted with the highest score (hence the .head)
        if(tasks(i).name == headTaskName) {
          heads = Some(tokenLabels.map(_.head._1.toInt))
        }
      }
    }

    // generate outputs for the dual tasks, if heads were predicted by one of the primary tasks
    if(heads.isDefined) {
      //println("Tokens:    " + tokens.mkString(", "))
      //println("Heads:     " + heads.get.mkString(", "))
      //println("Masks:     " + TokenClassifier.mkTokenMask(wordIds).mkString(", "))
      val masks = Some(TokenClassifier.mkTokenMask(wordIds))

      for(i <- tasks.indices) {
        if(tasks(i).dual) {
          val tokenLabels = tasks(i).predictWithScores(encOutput, heads, masks)
          val wordLabels = TokenClassifier.mapTokenLabelsAndScoresToWords(tokenLabels, tokenization.wordIds)
          allLabels(i) = wordLabels
        }
      }
    }

    allLabels
  }

  /** 
   * Predict labels for all tasks for a given sentence 
   * @param words Words in this sentence
   * @return Sequnce of labels for each task, for each token
   */
  def predict(words: Seq[String], headTaskName:String = "Deps Head"): Array[Array[String]] = {
    // tokenize to subword tokens
    val tokenization = LongTokenization(tokenizer.tokenize(words.toArray))
    val inputIds = tokenization.tokenIds
    val wordIds = tokenization.wordIds
    val tokens = tokenization.tokens

    // run the sentence through the transformer encoder
    val encOutput = encoder.forward(inputIds)

    // outputs for all tasks stored here
    val allLabels = new Array[Array[String]](tasks.length)
    var heads: Option[Array[Int]] = None

    // now generate token label predictions for all primary tasks (not dual!)
    for(i <- tasks.indices) {
      if(! tasks(i).dual) {
        val tokenLabels = tasks(i).predict(encOutput, None, None)
        val wordLabels = TokenClassifier.mapTokenLabelsToWords(tokenLabels, tokenization.wordIds)
        allLabels(i) = wordLabels

        // if this is the task that predicts head positions, then save them for the dual tasks
        if(tasks(i).name == headTaskName) {
          heads = Some(tokenLabels.map(_.toInt))
        }
      }
    }

    // generate outputs for the dual tasks, if heads were predicted by one of the primary tasks
    if(heads.isDefined) {
      //println("Tokens:    " + tokens.mkString(", "))
      //println("Heads:     " + heads.get.mkString(", "))
      //println("Masks:     " + TokenClassifier.mkTokenMask(wordIds).mkString(", "))
      val masks = Some(TokenClassifier.mkTokenMask(wordIds))

      for(i <- tasks.indices) {
        if(tasks(i).dual) {
          val tokenLabels = tasks(i).predict(encOutput, heads, masks)
          val wordLabels = TokenClassifier.mapTokenLabelsToWords(tokenLabels, tokenization.wordIds)
          allLabels(i) = wordLabels
        }
      }
    }

    allLabels
  }
}

object TokenClassifier {

  def fromFiles(modelDir: String): TokenClassifier = {
    val tokenClassifierLayout = new TokenClassifierLayout(modelDir)
    val tokenClassifierFactory = new TokenClassifierFactoryFromFiles(tokenClassifierLayout)

    tokenClassifierFactory.newTokenClassifier
  }

  def fromResources(modelDir: String): TokenClassifier = {
    val tokenClassifierLayout = new TokenClassifierLayout(modelDir)
    val tokenClassifierFactory = new TokenClassifierFactoryFromResources(tokenClassifierLayout)

    tokenClassifierFactory.newTokenClassifier
  }

  def mkTokenMask(wordIds: Array[Long]): Array[Boolean] = {
    wordIds.zipWithIndex.map{ case (wordId, index) =>
      mkSingleTokenMask(wordId, index, wordIds)  
    }
  }

  def mkSingleTokenMask(wordId: Long, index: Int, wordIds: Array[Long]): Boolean = {
    ! (wordId >= 0 && (index == 0 || wordId != wordIds(index - 1)))
  }

  def mapTokenLabelsToWords(tokenLabels: Array[String], wordIds: Array[Long]): Array[String] = {
    require(tokenLabels.length == wordIds.length)
    val wordLabelOpts = tokenLabels.zip(wordIds).zipWithIndex.map { case ((tokenLabel, wordId), index) =>
      val masked = mkSingleTokenMask(wordId, index, wordIds)
      if (! masked) Some(tokenLabel)
      else None
    }

    wordLabelOpts.flatten
  }

  def mapTokenLabelsAndScoresToWords(tokenLabels: Array[Array[(String, Float)]], wordIds: Array[Long]): Array[Array[(String, Float)]] = {
    require(tokenLabels.length == wordIds.length)
    val wordLabelOpts = tokenLabels.zip(wordIds).zipWithIndex.map { case ((tokenLabel, wordId), index) =>
      val masked = mkSingleTokenMask(wordId, index, wordIds)
      if (! masked) Some(tokenLabel)
      else None
    }

    wordLabelOpts.flatten
  }
}
