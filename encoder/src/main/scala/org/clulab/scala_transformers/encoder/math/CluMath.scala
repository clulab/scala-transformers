package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result

object CluMath extends Math {

  case class CluMatrix[T](rowCount: Int, colCount: Int, data: Array[Array[T]])

  sealed trait Orientation

  final class RowOrientation extends Orientation
  final class ColOrientation extends Orientation

  case class CluVector[R <: Orientation, T](count: Int, data: Array[T])

  type ColVector[T] = CluVector[ColOrientation, T]
  type RowVector[T] = CluVector[RowOrientation, T]

  type MathValue = Float

  type CluRowMatrix = CluMatrix[MathValue]
  type CluColVector = ColVector[MathValue]
  type CluRowVector = RowVector[MathValue]

  type MathRowMatrix = CluRowMatrix
  type MathColVector = CluColVector
  type MathRowVector = CluRowVector

  // TODO: How to get case class without the new?  Do I need the val thing?
  def fromResult(result: Result): Array[MathRowMatrix] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map { array2d =>
      val rows = array2d.length
      val cols = array2d.head.length
      val matrix = new CluRowMatrix(rows, cols, array2d)

      matrix
    }

    outputs
  }

  def argmax(rowVector: MathRowVector): Int = {
    val max = rowVector.data.max
    val index = rowVector.data.indexOf(max)

    index
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, colVector: MathColVector): Unit = {
    val colData = colVector.data

    matrix.data.foreach { row =>
      row.indices.foreach { colIndex =>
        row(colIndex) += colData(colIndex)
      }
    }
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, rowIndex: Int, rowVector: MathRowVector): Unit = {
    val matrixData = matrix.data(rowIndex)
    val vectorData = rowVector.data

    matrixData.indices.foreach { colIndex =>
      matrixData(colIndex) += vectorData(colIndex)
    }
  }

  def mul(leftMatrix: MathRowMatrix, rightMatrix: MathRowMatrix): MathRowMatrix = {
    require(leftMatrix.colCount == rightMatrix.rowCount)
    val rowCount = leftMatrix.rowCount
    val colCount = rightMatrix.colCount
    val leftData = leftMatrix.data
    val rightData = rightMatrix.data
    val data = Array.tabulate[MathValue](rowCount, colCount) { (rowIndex, colIndex) =>
      val leftRowData = leftData(rowIndex)
      val sum = leftRowData.indices.foldLeft(0f) { case (sum, index) =>
        sum + leftRowData(index) * rightData(index)(colIndex)
      }

      sum
    }

    new CluRowMatrix(rowCount, colCount, data)
  }

  def rows(matrix: MathRowMatrix): Int = {
    matrix.rowCount
  }

  def cols(matrix: MathRowMatrix): Int = {
    matrix.colCount
  }

  def length(colVector: MathColVector): Int = {
    colVector.count
  }

  // TODO: How often is this used?
  def t(matrix: MathRowMatrix): MathRowMatrix = {
    val rowCount = matrix.colCount
    val colCount = matrix.rowCount
    val data = Array.tabulate[MathValue](rowCount, colCount) { (rowIndex, colIndex) =>
      matrix.data(colIndex)(rowIndex)
    }

    new CluRowMatrix(rowCount, colCount, data)
  }

  def vertcat(leftColVector: MathColVector, rightColVector: MathColVector): MathColVector = {
    val data = {
      val data = new Array[MathValue](leftColVector.count + rightColVector.count)

      leftColVector.data.copyToArray(data, 0)
      rightColVector.data.copyToArray(data, leftColVector.count)
      data
    }

    new CluColVector(data.length, data)
  }

  def zeros(rows: Int, cols: Int): MathRowMatrix = {
    val data = Array.fill[Array[MathValue]](rows) { new Array(cols) }

    new CluRowMatrix(rows, cols, data)
  }

  def row(matrix: MathRowMatrix, index: Int): MathRowVector = {
    new CluRowVector(matrix.colCount, matrix.data(index).clone)
  }

  def horcat(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector = {
    val data = {
      val data = new Array[MathValue](leftRowVector.count + rightRowVector.count)

      leftRowVector.data.copyToArray(data, 0)
      rightRowVector.data.copyToArray(data, leftRowVector.count)
      data
    }

    new CluRowVector(data.length, data)
  }

  def toArray(rowVector: MathRowVector): Array[MathValue] = {
    rowVector.data
  }

  def get(rowVector: MathRowVector, index: Int): MathValue = {
    rowVector.data(index)
  }

  def mkMatrixFromRows(values: Array[Array[MathValue]]): MathRowMatrix = {
    new CluRowMatrix(values.length, values.head.length, values)
  }

  def mkMatrixFromCols(values: Array[Array[MathValue]]): MathRowMatrix = {
    val rows = values.length
    val cols = values.head.length
    val data = Array.fill[Array[MathValue]](cols) { new Array[MathValue](rows) }

    values.indices.foreach { rowIndex =>
      values(rowIndex).indices.foreach { colIndex =>
        data(colIndex)(rowIndex) = values(rowIndex)(colIndex)
      }
    }
    new CluRowMatrix(cols, rows, data)
  }

  def mkColVector(values: Array[MathValue]): MathColVector = {
    new CluColVector(values.length, values)
  }
}
