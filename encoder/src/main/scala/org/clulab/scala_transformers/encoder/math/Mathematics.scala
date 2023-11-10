package org.clulab.scala_transformers.encoder.math

object Mathematics {
  // Pick one of these.
  val Math = BreezeMath
//  val Math = EjmlMath

  type MathMatrix = Math.MathRowMatrix
  type MathColVector = Math.MathColVector
  type MathRowVector = Math.MathRowVector
}
