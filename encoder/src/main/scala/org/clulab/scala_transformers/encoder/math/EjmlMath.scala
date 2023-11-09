package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession.Result
import org.ejml.data.FMatrixRMaj
import org.ejml.simple.SimpleMatrix

class EjmlMath {

}

object EjmlMath {
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
    ???
  }

  def inplaceMatrixAddition(matrix: FMatrixRMaj, rowIndex: Int, vector: FMatrixRMaj): Unit = {
    ???
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
    vector.getNumCols
  }

  def t(matrix: FMatrixRMaj): FMatrixRMaj = {
    val result = SimpleMatrix.wrap(matrix).transpose().getMatrix[FMatrixRMaj]

    result
  }

  def vertcat(left: FMatrixRMaj, right: FMatrixRMaj): FMatrixRMaj = {
    val leftSimple = SimpleMatrix.wrap(left)
    val rightSimple = SimpleMatrix.wrap(right)
    val result = leftSimple.concatColumns(rightSimple).getMatrix[FMatrixRMaj]

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
