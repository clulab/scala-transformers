package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import org.ejml.data.FMatrixRMaj

class EjmlMath {

}

object EjmlMath {
  def fromResult(result: Result): Array[FMatrixRMaj] = {
    val array = result.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = array.map(new FMatrixRMaj(_)) // FMatrixRMaj

    outputs
  }

  def argmax(row: FMatrixRMaj): Int = {
    row.
  }

  def inplaceMatrixAddition(matrix: OnnxTensor, vector: OnnxTensor): Unit = {
    ???
  }

  def inplaceMatrixAddition(matrix: OnnxTensor, rowIndex: Int, vector: OnnxTensor): Unit = {
    ???
  }

  def mul(left: SimpleMatrix, right: SimpleMatrix): SimpleMatrix = {

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
    matrix.transpose
  }

  def vertcat(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def zeros(rows: Int, cols: Int): FMatrixRMaj = {
    new FMatrixRMaj(rows, cols)
  }

  def row(matrix: FMatrixRMaj, index: Int): FMatrixRMaj = {
    matrix.rows(index, index + 1)
  }

  def cat(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def toArray(vector: FMatrixRMaj): Array[Float] = {
    val result = vector.getData

    result
  }

  def get(vector: FMatrixRMaj, index: Int): Float = {
    vector.get(0, index)
  }

  def mkRowMatrix(values: Array[Array[Float]]): FMatrixRMaj = {
    new FMatrixRMaj(values)
  }

  def mkVector(values: Array[Float]): FMatrixRMaj = {
    new FMatrixRMaj(values)
  }
}
