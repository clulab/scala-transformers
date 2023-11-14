package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import org.apache.commons.math3.linear.{Array2DRowRealMatrix, ArrayRealVector, DefaultRealMatrixChangingVisitor}

object CommonsMath extends Math {
  type MathValue = Float
  type MathRowMatrix = Array2DRowRealMatrix
  type MathColVector = ArrayRealVector
  type MathRowVector = ArrayRealVector

  def fromResult(result: Result): Array[MathRowMatrix] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map { array2d =>
      val rows = array2d.length
      val cols = array2d.head.length
      val matrix = new Array2DRowRealMatrix(rows, cols)

      for (row <- 0 until rows)
        for (col <- 0 until cols)
          matrix.setEntry(row, col, array2d(row)(col).toDouble)
       matrix
    }

    outputs
  }

  def argmax(rowVector: MathRowVector): Int = {
    rowVector.getMaxIndex
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, colVector: MathColVector): Unit = {
    val visitor = new DefaultRealMatrixChangingVisitor {
      override def visit(row: Int, column: Int, value: Double): Double = {
        value + colVector.getEntry(column)
      }
    }

    matrix.walkInRowOrder(visitor, 0, matrix.getRowDimension - 1, 0, matrix.getColumnDimension - 1)
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, rowIndex: Int, rowVector: MathRowVector): Unit = {
    val visitor = new DefaultRealMatrixChangingVisitor {
      override def visit(row: Int, column: Int, value: Double): Double = {
        value + rowVector.getEntry(column)
      }
    }

    matrix.walkInRowOrder(visitor, rowIndex, rowIndex, 0, matrix.getColumnDimension - 1)
  }

  def mul(leftMatrix: MathRowMatrix, rightMatrix: MathRowMatrix): MathRowMatrix = {
    leftMatrix.multiply(rightMatrix)
  }

  def rows(matrix: MathRowMatrix): Int = {
    matrix.getRowDimension
  }

  def cols(matrix: MathRowMatrix): Int = {
    matrix.getColumnDimension
  }

  def length(colVector: MathColVector): Int = {
    colVector.getDimension
  }

  def vertcat(leftColVector: MathColVector, rightColVector: MathColVector): MathColVector = {
    new ArrayRealVector(leftColVector, rightColVector)
  }

  def zeros(rows: Int, cols: Int): MathRowMatrix = {
    new Array2DRowRealMatrix(rows, cols)
  }

  def row(matrix: MathRowMatrix, index: Int): MathRowVector = {
    new ArrayRealVector(matrix.getRow(index))
  }

  def horcat(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector = {
    new ArrayRealVector(leftRowVector, rightRowVector)
  }

  def toArray(rowVector: MathRowVector): Array[MathValue] = {
    val doubleArray = rowVector.toArray
    val array = doubleArray.map(_.toFloat)

    array
  }

  def get(rowVector: MathRowVector, index: Int): MathValue = {
    rowVector.getEntry(index).toFloat
  }

  def mkMatrixFromRows(values: Array[Array[Float]]): MathRowMatrix = {
    val rows = values.length
    val cols = values.head.length
    val matrix = new Array2DRowRealMatrix(rows, cols)

    for (row <- 0 until rows)
      for (col <- 0 until cols)
        matrix.setEntry(row, col, values(row)(col).toDouble)
    matrix
  }

  def mkMatrixFromCols(values: Array[Array[Float]]): MathRowMatrix = {
    val rows = values.length
    val cols = values.head.length
    val matrix = new Array2DRowRealMatrix(cols, rows)

    for (row <- 0 until rows)
      for (col <- 0 until cols)
        matrix.setEntry(col, row, values(row)(col).toDouble)
    matrix
  }

  // How do we keep track that this is a column?
  def mkColVector(values: Array[Float]): MathColVector = {
    val doubles = Array.tabulate[Double](values.length) { index =>
      values(index).toDouble
    }
    new ArrayRealVector(doubles)
  }
}
