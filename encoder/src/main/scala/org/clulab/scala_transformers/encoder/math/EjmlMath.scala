package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import org.ejml.data.FMatrixRMaj
import org.ejml.simple.SimpleMatrix

object EjmlMath extends Math {
  type MathValue = Float
  type MathRowMatrix = FMatrixRMaj
  type MathColVector = FMatrixRMaj
  type MathRowVector = FMatrixRMaj

  protected def isRowVector(rowVector: MathRowVector): Boolean = rowVector.getNumRows == 1

  protected def isColVector(colVector: MathColVector): Boolean = colVector.getNumCols == 1

  def fromResult(result: Result): Array[MathRowMatrix] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map(new FMatrixRMaj(_))

    outputs
  }

  def argmax(rowVector: MathRowVector): Int = {
    assert(isRowVector(rowVector))

    var maxIndex = 0
    var maxValue = rowVector.get(maxIndex)

    1.until(rowVector.getNumCols).foreach { index =>
      val value = rowVector.get(index)

      if (value > maxValue) {
        maxValue = value
        maxIndex = index
      }
    }

    maxIndex
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, colVector: MathColVector): Unit = {
    assert(isColVector(colVector))
    0.until(matrix.getNumRows) foreach { row =>
      0.until(matrix.getNumCols) foreach { col =>
        val oldVal = matrix.get(row, col)
        val newVal = oldVal + colVector.get(col, 0)

        matrix.set(row, col, newVal)
      }
    }
  }

  def inplaceMatrixAddition(matrix: MathRowMatrix, rowIndex: Int, rowVector: MathRowVector): Unit = {
    assert(isRowVector(rowVector))
    0.until(matrix.getNumCols) foreach { col =>
      val oldVal = matrix.get(rowIndex, col)
      val newVal = oldVal + rowVector.get(0, col)

      matrix.set(rowIndex, col, newVal)
    }
  }

  def mul(leftMatrix: MathRowMatrix, rightMatrix: MathRowMatrix): MathRowMatrix = {
    val leftSimple = SimpleMatrix.wrap(leftMatrix)
    val rightSimple = SimpleMatrix.wrap(rightMatrix)
    val product = leftSimple.mult(rightSimple)
    val matrix = product.getMatrix.asInstanceOf[FMatrixRMaj]

    matrix
  }

  def rows(matrix: MathRowMatrix): Int = {
    matrix.getNumRows
  }

  def cols(matrix: MathRowMatrix): Int = {
    matrix.getNumCols
  }

  def length(colVector: MathColVector): Int = {
    assert(isColVector(colVector))
    colVector.getNumRows
  }

  def t(matrix: MathRowMatrix): MathRowMatrix = {
    val result = SimpleMatrix.wrap(matrix).transpose().getMatrix[FMatrixRMaj]

    result
  }

  def vertcat(leftColVector: MathColVector, rightColVector: MathColVector): MathColVector = {
    assert(isColVector(leftColVector))
    assert(isColVector(rightColVector))
    val leftSimple = SimpleMatrix.wrap(leftColVector)
    val rightSimple = SimpleMatrix.wrap(rightColVector)
    val result = leftSimple.concatRows(rightSimple).getMatrix[FMatrixRMaj]

    assert(isColVector(result))
    result
  }

  def zeros(rows: Int, cols: Int): MathRowMatrix = {
    new FMatrixRMaj(rows, cols)
  }

  def row(matrix: MathRowMatrix, index: Int): MathRowVector = {
    val result = SimpleMatrix.wrap(matrix).rows(index, index + 1).getMatrix[FMatrixRMaj]

    assert(isRowVector(result))
    result
  }

  def cat(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector = {
    assert(isRowVector(leftRowVector))
    assert(isRowVector(rightRowVector))
    val leftSimple = SimpleMatrix.wrap(leftRowVector)
    val rightSimple = SimpleMatrix.wrap(rightRowVector)
    val result = leftSimple.concatColumns(rightSimple).getMatrix[FMatrixRMaj]

    assert(isRowVector(result))
    result
  }

  def toArray(rowVector: MathRowVector): Array[MathValue] = {
    assert(isRowVector(rowVector))
    val result = rowVector.getData

    result
  }

  def get(rowVector: MathRowVector, index: Int): MathValue = {
    assert(isRowVector(rowVector))
    rowVector.get(index)
  }

  def mkRowMatrix(values: Array[Array[MathValue]]): MathRowMatrix = {
    new FMatrixRMaj(values)
  }

  def mkVector(values: Array[MathValue]): MathColVector = {
    val result = new FMatrixRMaj(values)

    assert(isColVector(result))
    result
  }
}