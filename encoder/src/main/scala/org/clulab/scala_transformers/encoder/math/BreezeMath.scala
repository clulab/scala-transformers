package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import breeze.linalg.{DenseMatrix, DenseVector, Transpose, `*`, argmax => BreezeArgmax}
import org.clulab.scala_transformers.encoder.BreezeUtils

object BreezeMath extends Math {
  type MathValue = Float
  type MathRowMatrix = DenseMatrix[MathValue]
  type MathColVector = DenseVector[MathValue]
  type MathRowVector = Transpose[DenseVector[MathValue]]

  def fromResult(result: Result): Array[MathRowMatrix] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map(BreezeUtils.mkRowMatrix(_))

    outputs
  }

  def argmax(rowVector: MathRowVector): Int = {
    val bestIndex = BreezeArgmax(rowVector.t)

    bestIndex
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, colVector: MathColVector): Unit = {
    matrix(*, ::) :+= colVector
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, rowIndex: Int, rowVector: MathRowVector): Unit = {

    matrix(rowIndex, ::) :+= rowVector
  }

  def mul(leftMatrix: MathRowMatrix, rightMatrix: MathRowMatrix): MathRowMatrix = {
    leftMatrix * rightMatrix
  }

  def rows(matrix: MathRowMatrix): Int = {
    matrix.rows
  }

  def cols(matrix: MathRowMatrix): Int = {
    matrix.cols
  }

  def length(colVector: MathColVector): Int = {
    colVector.length
  }

  def t(matrix: MathRowMatrix): MathRowMatrix = {
    matrix.t
  }

  def vertcat(leftColVector: MathColVector, rightColVector: MathColVector): MathColVector = {
    DenseVector.vertcat(leftColVector, rightColVector)
  }

  def zeros(rows: Int, cols: Int): MathRowMatrix = {
    DenseMatrix.zeros[Float](rows, cols)
  }

  def row(matrix: MathRowMatrix, index: Int): MathRowVector = {
    matrix(index, ::)
  }

  def cat(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector = {
    DenseVector.vertcat(leftRowVector.t, rightRowVector.t).t
  }

  def toArray(rowVector: MathRowVector): Array[MathValue] = {
    rowVector.t.toArray
  }

  def get(rowVector: MathRowVector, index: Int): MathValue = {
    rowVector(index)
  }

  def mkRowMatrix(values: Array[Array[MathValue]]): MathRowMatrix = {
    BreezeUtils.mkRowMatrix(values)
  }

  def mkVector(values: Array[MathValue]): MathColVector = {
    DenseVector(values)
  }
}
