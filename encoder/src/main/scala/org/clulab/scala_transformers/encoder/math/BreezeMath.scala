package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import breeze.linalg.{DenseMatrix, DenseVector, Transpose, `*`, argmax => BreezeArgmax}
import org.clulab.scala_transformers.encoder.BreezeUtils

class BreezeMath {
}

object BreezeMath {

  def fromResult(result: Result): Array[DenseMatrix[Float]] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map(BreezeUtils.mkRowMatrix(_))

    outputs
  }

  def argmax(row: DenseVector[Float]): Int = {
    val bestIndex = BreezeArgmax(row.t)

    bestIndex
  }

  def inplaceAddition(matrix: DenseMatrix[Float], b: DenseVector[Float]): Unit = {
    matrix(*, ::) :+= b
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
}
