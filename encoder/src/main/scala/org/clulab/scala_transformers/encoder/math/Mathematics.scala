package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OnnxTensor
import breeze.linalg.{DenseMatrix, DenseVector, Transpose}
import org.ejml.data.FMatrixRMaj

object Mathematics {
  // Pick one of these.
  val Math = BreezeMath // This takes care of the companion object.
//  val Math = EjmlMath
//  val Math = OnnxMath

  type MathMatrix = Math.MathMatrix
  type MathVector = Math.MathVector
  type MathRowVector = Math.MathRowVector
}
