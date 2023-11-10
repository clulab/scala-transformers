package org.clulab.scala_transformers.encoder.math

object Mathematics {
  // Pick one of these.
  val Math = BreezeMath // This takes care of the companion object.
//  val Math = EjmlMath
//  val Math = OnnxMath

  type MathMatrix = Math.MathRowMatrix
  type MathColVector = Math.MathColVector
  type MathRowVector = Math.MathRowVector
}
