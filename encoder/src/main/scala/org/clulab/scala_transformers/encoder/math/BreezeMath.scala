package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import breeze.linalg.{DenseMatrix, DenseVector, Transpose, `*`, argmax => BreezeArgmax}
import org.clulab.scala_transformers.encoder.BreezeUtils

object BreezeMath {
  type MathValue = Float
  type MathMatrix = DenseMatrix[MathValue]
  type MathVector = DenseVector[MathValue]
  type MathRowVector = Transpose[DenseVector[MathValue]]

  def fromResult(result: Result): Array[DenseMatrix[Float]] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map(BreezeUtils.mkRowMatrix(_))

    outputs
  }

  def argmax(row: Transpose[DenseVector[Float]]): Int = {
    val bestIndex = BreezeArgmax(row.t)

    bestIndex
  }

  def inplaceMatrixAddition(matrix: DenseMatrix[Float], vector: DenseVector[Float]): Unit = {
    matrix(*, ::) :+= vector
  }

  def inplaceMatrixAddition(matrix: DenseMatrix[Float], rowIndex: Int, vector: Transpose[DenseVector[Float]]): Unit = {

    matrix(rowIndex, ::) :+= vector
  }

  def mul(left: DenseMatrix[Float], right: DenseMatrix[Float]): DenseMatrix[Float] = {
    left * right
  }

  def rows(matrix: DenseMatrix[Float]): Int = {
    matrix.rows
  }

  def cols(matrix: DenseMatrix[Float]): Int = {
    matrix.cols
  }

  def length(vector: DenseVector[Float]): Int = {
    vector.length
  }

  def t(matrix: DenseMatrix[Float]): DenseMatrix[Float] = {
    matrix.t
  }

  def vertcat(left: DenseVector[Float], right: DenseVector[Float]): DenseVector[Float] = {
    DenseVector.vertcat(left, right)
  }

  def zeros(rows: Int, cols: Int): DenseMatrix[Float] = {
    DenseMatrix.zeros[Float](rows, cols)
  }

  def row(matrix: DenseMatrix[Float], index: Int): Transpose[DenseVector[Float]] = {
    matrix(index, ::)
  }

  def cat(left: Transpose[DenseVector[Float]], right: Transpose[DenseVector[Float]]): Transpose[DenseVector[Float]] = {
    DenseVector.vertcat(left.t, right.t).t
  }

  def toArray(vector: Transpose[DenseVector[Float]]): Array[Float] = {
    vector.t.toArray
  }

  def get(vector: Transpose[DenseVector[Float]], index: Int): Float = {
    vector(index)
  }

  def mkRowMatrix(values: Array[Array[Float]]): DenseMatrix[Float] = {
    BreezeUtils.mkRowMatrix(values)
  }

  def mkVector(values: Array[Float]): DenseVector[Float] = {
    DenseVector(values)
  }
}
