package org.clulab.scala_transformers.apps

import org.clulab.scala_transformers.encoder.TokenClassifier

import java.io.PrintWriter

/** Processes a file in the CoNLL format and outputs BIO notations in the format expected by the CoNLL scorer */
object TokenClassifierCoNLLApp extends App {
  val WORD_POS = 0 // in the column format
  val TAG_POS = 1 // in the column format
  val ANNOTATION_POS = 0 // in the encoder MTL outputs

  val fileName = "test.txt"
  val pw = new PrintWriter(fileName + ".conll")

  val tokenClassifier = TokenClassifier.fromFiles("../microsoft-deberta-v3-base-mtl/avg_export")

  pw.close()
}
