package org.clulab.scala_transformers.encoder

/** Couples one encoder with one or more linear layers (one per task) */
class TokenClassifier(val encoder: Encoder, val tasks: Map[String, LinearLayer]) {

}

object TokenClassifier {
  
}