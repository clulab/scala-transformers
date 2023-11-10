package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import breeze.linalg.{DenseMatrix, DenseVector, Transpose, `*`, argmax => BreezeArgmax}

object BreezeMath extends Math {
  type MathValue = Float
  type MathRowMatrix = DenseMatrix[MathValue]
  type MathColVector = DenseVector[MathValue]
  type MathRowVector = Transpose[DenseVector[MathValue]]

  def fromResult(result: Result): Array[MathRowMatrix] = {
    val arrays = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = arrays.map(mkMatrixFromRows(_))

    outputs
  }

  def argmax(rowVector: MathRowVector): Int = {
    BreezeArgmax(rowVector)
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

  def horcat(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector = {
    DenseVector.vertcat(leftRowVector.t, rightRowVector.t).t
  }

  def toArray(rowVector: MathRowVector): Array[MathValue] = {
    rowVector.t.toArray
  }

  def get(rowVector: MathRowVector, index: Int): MathValue = {
    rowVector(index)
  }

  def mkMatrixFromRows(values: Array[Array[MathValue]]): MathRowMatrix = {
    val rows = values.length
    val cols = values.head.length
    val denseMatrix = new DenseMatrix[Float](rows, cols)

    for (row <- 0 until rows)
      for (col <- 0 until cols)
        denseMatrix(row, col) = values(row)(col)
    denseMatrix
  }

  def mkMatrixFromCols(values: Array[Array[MathValue]]): MathRowMatrix = {
    val rows = values.length
    val cols = values.head.length
    val denseMatrix = new DenseMatrix[Float](cols, rows)

    for (row <- 0 until rows)
      for (col <- 0 until cols)
        denseMatrix(col, row) = values(row)(col)
    denseMatrix
  }

  def mkColVector(values: Array[MathValue]): MathColVector = {
    DenseVector(values)
  }
}
