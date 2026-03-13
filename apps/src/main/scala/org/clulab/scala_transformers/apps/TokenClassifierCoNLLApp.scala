package org.clulab.scala_transformers.apps

import org.clulab.scala_transformers.encoder.TokenClassifier

import java.io.PrintWriter

/** Processes a file in the CoNLL format and outputs BIO notations in the format expected by the CoNLL scorer */
object TokenClassifierCoNLLApp extends App {
  val WORD_POS = 0 // in the column format
  val LABEL_POS = 1 // in the column format
  val ANNOTATION_POS = 0 // in the encoder MTL outputs

  val fileName = "../test.txt"
  val pw = new PrintWriter(fileName + ".conll")

  val tokenClassifier = TokenClassifier.fromFiles("../answerdotai-ModernBERT-base-mtl/avg_export") // "../microsoft-deberta-v3-base-mtl/avg_export")
  val goldDoc = ColumnsToDocument.readFromFile(fileName, WORD_POS, LABEL_POS)
  val sentences = goldDoc.sentences
  println(s"Read a doc with ${sentences.length} sentences.")

  for(i <- sentences.indices) {
    val words = sentences(i).words
    val goldLabels = sentences(i).labels

    val predLabels = tokenClassifier.predict(words)(ANNOTATION_POS)

    println("words: " + words.mkString(", "))
    println("gold:  " + goldLabels.mkString(", "))
    println("pred:  " + predLabels.mkString(", "))

    if(goldLabels.length == predLabels.length) {
      for (j <- words.indices) {
        pw.println(s"${words(j)} ${goldLabels(j)} ${predLabels(j)}")
      }
      pw.println()
    }
  }

  pw.close()
}
