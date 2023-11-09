package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OnnxTensor
import breeze.linalg.{DenseMatrix, DenseVector}
import org.ejml.data.FMatrixRMaj

object Mathematics {
  type MathValue = Float

//  type Math = BreezeMath // This takes care of the class.
//  val Math = BreezeMath // This takes care of the companion object.
//
//  type MathMatrix = DenseMatrix[MathValue]
//  val MathMatrix = DenseMatrix
//
//  type MathVector = DenseVector[MathValue]
//  val MathVector = DenseVector

//  type Math = OnnxMath // This takes care of the class.
//  val Math = OnnxMath // This takes care of the companion object.
//
//  type MathMatrix = OnnxTensor
//  type MathVector = OnnxTensor

  type Math = EjmlMath
  val Math = EjmlMath

  type MathMatrix = FMatrixRMaj
  type MathVector = FMatrixRMaj
}
