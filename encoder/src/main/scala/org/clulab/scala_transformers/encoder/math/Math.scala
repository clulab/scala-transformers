package org.clulab.scala_transformers.encoder.math

import breeze.linalg.{DenseMatrix, DenseVector}

object Math {
  type MathValue = Float

  type Mathematician = BreezeMath // This takes care of the class.
  val Mathematician = BreezeMath // This takes care of the companion object.

  type MathMatrix = DenseMatrix[MathValue]
  val MathMatrix = DenseMatrix

  type MathVector = DenseVector[MathValue]
  val MathVector = DenseVector

//  type Mathematician = OnnxMath // This takes care of the class.
//  val Mathematician = OnnxMath // This takes care of the companion object.
}
