package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import org.ejml.data.FMatrixRMaj
import org.ejml.simple.SimpleMatrix

object EjmlMath {
  type MathMatrix = FMatrixRMaj
  type MathVector = FMatrixRMaj
  type MathRowVector = FMatrixRMaj

  def fromResult(result: Result): Array[FMatrixRMaj] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map(new FMatrixRMaj(_)) // FMatrixRMaj

    outputs
  }

  def argmax(row: FMatrixRMaj): Int = {
    var maxIndex = 0
    var maxValue = row.get(maxIndex)

    1.until(row.getNumCols).foreach { index =>
      val value = row.get(index)

      if (value > maxValue) {
        maxValue = value
        maxIndex = index
      }
    }

    maxIndex
  }

  def inplaceMatrixAddition(matrix: FMatrixRMaj, vector: FMatrixRMaj): Unit = {
    0.until(matrix.getNumRows) foreach { row =>
      0.until(matrix.getNumCols) foreach { col =>
        val oldVal = matrix.get(row, col)
        val newVal = oldVal + vector.get(col, 0)

        matrix.set(row, col, newVal)
      }
    }
  }

  def inplaceMatrixAddition(matrix: FMatrixRMaj, rowIndex: Int, vector: FMatrixRMaj): Unit = {
    0.until(matrix.getNumCols) foreach { col =>
      val oldVal = matrix.get(rowIndex, col)
      val newVal = oldVal + vector.get(0, col)

      matrix.set(rowIndex, col, newVal)
    }
  }

  def mul(left: FMatrixRMaj, right: FMatrixRMaj): FMatrixRMaj = {
    val leftSimple = SimpleMatrix.wrap(left)
    val rightSimple = SimpleMatrix.wrap(right)
    val product = leftSimple.mult(rightSimple)
    val matrix = product.getMatrix.asInstanceOf[FMatrixRMaj]

    matrix
  }

  def rows(matrix: FMatrixRMaj): Int = {
    matrix.getNumRows
  }

  def cols(matrix: FMatrixRMaj): Int = {
    matrix.getNumCols
  }

  def length(vector: FMatrixRMaj): Int = {
    // This will be a vertical vector.
    vector.getNumRows
  }

  def t(matrix: FMatrixRMaj): FMatrixRMaj = {
    val result = SimpleMatrix.wrap(matrix).transpose().getMatrix[FMatrixRMaj]

    result
  }

  def vertcat(left: FMatrixRMaj, right: FMatrixRMaj): FMatrixRMaj = {
    val leftSimple = SimpleMatrix.wrap(left)
    val rightSimple = SimpleMatrix.wrap(right)
    val result = leftSimple.concatRows(rightSimple).getMatrix[FMatrixRMaj]

    result
  }

  def zeros(rows: Int, cols: Int): FMatrixRMaj = {
    new FMatrixRMaj(rows, cols)
  }

  def row(matrix: FMatrixRMaj, index: Int): FMatrixRMaj = {
    val result = SimpleMatrix.wrap(matrix).rows(index, index + 1).getMatrix[FMatrixRMaj]

    result
  }

  def cat(left: FMatrixRMaj, right: FMatrixRMaj): FMatrixRMaj = {
    val leftSimple = SimpleMatrix.wrap(left)
    val rightSimple = SimpleMatrix.wrap(right)
    val result = leftSimple.concatRows(rightSimple).getMatrix[FMatrixRMaj]

    result
  }

  def toArray(vector: FMatrixRMaj): Array[Float] = {
    val result = vector.getData

    result
  }

  def get(vector: FMatrixRMaj, index: Int): Float = {
    vector.get(index)
  }

  def mkRowMatrix(values: Array[Array[Float]]): FMatrixRMaj = {
    new FMatrixRMaj(values)
  }

  def mkVector(values: Array[Float]): FMatrixRMaj = {
    new FMatrixRMaj(values)
  }
}
