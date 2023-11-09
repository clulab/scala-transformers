package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.encoder.math.Mathematics.{MathMatrix, MathValue, MathVector, Math}

/** Implements one linear layer */
class LinearLayer(
  val name: String,
  val dual: Boolean, 
  val weights: MathMatrix, // dimensions (hidden state size x labels size)
  val biasesOpt: Option[MathVector], // column vector with length = labels size
  val labelsOpt: Option[Array[String]]
) {

  /** Forward pass for a single sentence */
  def forward(inputSentence: MathMatrix): MathMatrix = {
    val batch = Array(inputSentence)
    forward(batch).head
  }

  /**
   * Forward pass for a batch of sentences
   * @param inputBatch Each matrix in the batch has dimensions (sentence size x hidden state size)
   * @return Each output matrix has dimensions (sentence size x labels size)
   */
  def forward(inputBatch: Array[MathMatrix]): Array[MathMatrix] = {
    inputBatch.map { input =>
      //println("INPUT:\n" + input)
      val output = Math.mul(input, weights)
      //println("OUTPUT before bias:\n" + output)
      for (b <- biasesOpt) Math.inplaceMatrixAddition(output, b)
      //println("OUTPUT after bias:\n" + output)
      output
    }
  }

  /** Predict the top label per token */
  def predict(inputSentence: MathMatrix,
              heads: Option[Array[Int]], 
              masks: Option[Array[Boolean]]): Array[String] = {
    val batchSentences = Array(inputSentence)
    val batchHeads = heads.map(Array(_))
    val batchMasks = masks.map(Array(_))
    predict(batchSentences, batchHeads, batchMasks).head
  }

  /** Predict the top label for each token in each sentence in the batch */
  def predict(inputBatch: Array[MathMatrix],
              batchHeads: Option[Array[Array[Int]]],
              batchMasks: Option[Array[Array[Boolean]]]): Array[Array[String]] = {
    if (dual) predictDual(inputBatch, batchHeads, batchMasks)
    else predictPrimal(inputBatch)
  }

  /** Predict all labels and their scores per token */
  def predictWithScores(inputSentence: MathMatrix,
                        heads: Option[Array[Array[Int]]], 
                        masks: Option[Array[Boolean]]): Array[Array[(String, Float)]] = {
    val batchSentences = Array(inputSentence)
    val batchHeads = heads.map(Array(_))
    val batchMasks = masks.map(Array(_))
    predictWithScores(batchSentences, batchHeads, batchMasks).head
  }

  /** Predict all labels and their scores per token in each sentence in the batch */
  def predictWithScores(inputBatch: Array[MathMatrix],
                        batchHeads: Option[Array[Array[Array[Int]]]],
                        batchMasks: Option[Array[Array[Boolean]]]): Array[Array[Array[(String, Float)]]] = {
    if (dual) predictDualWithScores(inputBatch, batchHeads, batchMasks)
    else predictPrimalWithScores(inputBatch)
  }

  def concatenateModifiersAndHeads(
    sentenceHiddenStates: MathMatrix,
    headRelativePositions: Array[Int]): MathMatrix = {

    // this matrix concatenates the hidden states of modifier + corresponding head
    // rows = number of tokens in the sentence; cols = hidden state size x 2
    val concatMatrix = Math.zeros(rows = Math.rows(sentenceHiddenStates), cols = 2 * Math.cols(sentenceHiddenStates))

    // traverse all modifiers
    for(i <- 0 until Math.rows(sentenceHiddenStates)) {
      val modHiddenState = Math.row(sentenceHiddenStates, i)
      // what is the absolute position of the head token in the sentence?
      val rawHeadAbsPos = i + headRelativePositions(i)
      val headAbsolutePosition = 
        if(rawHeadAbsPos >= 0 && rawHeadAbsPos < Math.rows(sentenceHiddenStates)) rawHeadAbsPos
        else i // if the absolute position is invalid (e.g., root node or incorrect prediction) duplicate the mod embedding
      val headHiddenState = Math.row(sentenceHiddenStates, headAbsolutePosition)

      // vector concatenation in Breeze operates over vertical vectors, hence the transposing here
      val concatState = Math.cat(modHiddenState, headHiddenState)
      // row i in the concatenated matrix contains the embedding of modifier i and its head
      Math.inplaceVectorAddition(Math.row(concatMatrix, i), concatState)
    }
    
    concatMatrix
  }

  /**
    * Generates a 1-row matrix containing a concatenation of the modifier and head embeddings
    *
    */
  def concatenateModifierAndHead(
    sentenceHiddenStates: MathMatrix,
    modifierAbsolutePosition: Int,
    headRelativePosition: Int): MathMatrix = {

    // this matrix concatenates the hidden states of modifier + corresponding head
    // rows = 1; cols = hidden state size x 2
    val concatMatrix = Math.zeros(rows = 1, cols = 2 * Math.cols(sentenceHiddenStates))

    // embedding of the modifier
    val modHiddenState = Math.row(sentenceHiddenStates, modifierAbsolutePosition)

    // embedding of the head
    val rawHeadAbsPos = modifierAbsolutePosition + headRelativePosition
    val headAbsolutePosition = 
      if(rawHeadAbsPos >= 0 && rawHeadAbsPos < Math.rows(sentenceHiddenStates)) rawHeadAbsPos
      else modifierAbsolutePosition // if the absolute position is invalid (e.g., root node or incorrect prediction) duplicate the mod embedding
    val headHiddenState = Math.row(sentenceHiddenStates, headAbsolutePosition)

    // concatenation of the modifier and head embeddings
    // vector concatenation in Breeze operates over vertical vectors, hence the transposing here
    val concatState = Math.cat(modHiddenState, headHiddenState)
    val row = Math.row(concatMatrix, 0)
    Math.inplaceVectorAddition(row, concatState)
    concatMatrix
  }

  /** Predict the top label for each combination of modifier token and corresponding head token */
  def predictDual(inputBatch: Array[MathMatrix],
                  batchHeads: Option[Array[Array[Int]]] = None,
                  batchMasks: Option[Array[Array[Boolean]]] = None): Array[Array[String]] = {
    assert(batchHeads.isDefined)
    assert(batchMasks.isDefined)
    val indexToLabel = labelsOpt.getOrElse(throw new RuntimeException("ERROR: can't predict without labels!"))

    // we process one sentence at a time because the dual setting makes it harder to batch
    val outputBatch = inputBatch.zip(batchHeads.get).map { case (input, heads) =>
      // generate a matrix that is twice as wide to concatenate the embeddings of the mod + head
      val concatInput = concatenateModifiersAndHeads(input, heads)
      // get the logits for the current sentence produced by this linear layer
      val logitsPerSentence = forward(Array(concatInput))(0)
      // one token per row; pick argmax per token
      val bestLabels = Range(0, Math.rows(logitsPerSentence)).map { i =>
        val row = Math.row(logitsPerSentence, i) // picks line i from a 2D matrix
        val bestIndex = Math.argmax(row)

        indexToLabel(bestIndex)
      }

      bestLabels.toArray
    }

    outputBatch                    
  }

  // predicts the top label for each of the candidate heads
  // out dimensions: sentence in batch x token in sentence x label/score per token
  // batchHeads dimensions: sentence in batch x token in sentence x heads per token
  // labels are sorted in descending order of their scores
  def predictDualWithScores(inputBatch: Array[MathMatrix],
                            batchHeads: Option[Array[Array[Array[Int]]]] = None,
                            batchMasks: Option[Array[Array[Boolean]]] = None): Array[Array[Array[(String, Float)]]] = {
    assert(batchHeads.isDefined)
    assert(batchMasks.isDefined)
    val indexToLabel = labelsOpt.getOrElse(throw new RuntimeException("ERROR: can't predict without labels!"))

    // dimensions: sent in batch x token in sentence x label per candidate head
    // we process one sentence at a time because the dual setting makes it harder to batch
    val outputBatch = inputBatch.zip(batchHeads.get).map { case (input, headCandidatesPerSentence) =>
      // now process each token separately
      headCandidatesPerSentence.zipWithIndex.map { case (headCandidatesPerToken, modifierAbsolutePosition) =>
        // process each head candidate for this token
        headCandidatesPerToken.map { headRelativePosition =>
          // generate a matrix that is twice as wide to concatenate the embeddings of the mod + head
          val concatInput = concatenateModifierAndHead(input, modifierAbsolutePosition, headRelativePosition)
          // get the logits for the current pair of modifier and head
          val logitsPerSentence = forward(Array(concatInput))(0)
          val labelScores = Math.row(logitsPerSentence, 0)
          val bestIndex = Math.argmax(labelScores)
          val bestScore = Math.get(labelScores, bestIndex)
          val bestLabel = indexToLabel(bestIndex)

          // println(s"Top prediction for mod $modifierAbsolutePosition and relative head $headRelativePosition is $bestLabel with score $bestScore")
          (bestLabel, bestScore)
        } // end head candidates for this token
      } // end this token
    } // end sentence batch

    outputBatch                    
  }

  def predictPrimal(inputBatch: Array[MathMatrix]): Array[Array[String]] = {
    val labels = labelsOpt.getOrElse(throw new RuntimeException("ERROR: can't predict without labels!"))
    // predict best label per (subword) token
    val logits = forward(inputBatch)
    val outputBatch = logits.map { logitsPerSentence =>
      // one token per row; pick argmax per token
      val bestLabels = Range(0, Math.rows(logitsPerSentence)).map { i =>
        val row = Math.row(logitsPerSentence, i) // picks line i from a 2D matrix
        val bestIndex = Math.argmax(row)

        labels(bestIndex)
      }

      bestLabels.toArray
    }

    outputBatch
  }

  // out dimensions: sentence in batch x token in sentence x label/score per token
  // labels are sorted in descending order of their scores
  def predictPrimalWithScores(inputBatch: Array[MathMatrix]): Array[Array[Array[(String, Float)]]] = {
    val labels = labelsOpt.getOrElse(throw new RuntimeException("ERROR: can't predict without labels!"))
    // predict best label per (subword) token
    val logits = forward(inputBatch)
    val outputBatch = logits.map { logitsPerSentence =>
      // one token per row; store scores for all labels for this token
      val allLabels = Range(0, Math.rows(logitsPerSentence)).map { i =>
        // picks line i from a 2D matrix and converts it to Array
        val scores = Math.toArray(Math.row(logitsPerSentence, i))
        val labelsAndScores = labels.zip(scores)

        // keep scores in descending order (largest first) 
        labelsAndScores.sortBy(- _._2) // - score guarantees sorting in descending order of scores
      }

      allLabels.toArray
    }

    outputBatch
  }
}

object LinearLayer {

  def fromFiles(layerDir: String): LinearLayer = {
    val linearLayerLayout = new LinearLayerLayout(layerDir)
    val linearLayerFactory = new LinearLayerFactoryFromFiles(linearLayerLayout)

    linearLayerFactory.newLinearLayer
  }

  def fromResources(layerDir: String): LinearLayer = {
    val linearLayerLayout = new LinearLayerLayout (layerDir)
    val linearLayerFactory = new LinearLayerFactoryFromResources(linearLayerLayout)

    linearLayerFactory.newLinearLayer
  }
}
