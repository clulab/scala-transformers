package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession.Result

class OnnxMath {
}

object OnnxMath {

  // This is 1 x 72 x 768
  // Turn it into an Array[OnnxTensor] with the tne tensor at 72 x 768
  def fromResult(result: Result): Array[OnnxTensor] = {
    val outputs = result.get("sequence_output").asInstanceOf[OnnxTensor]

    Array(outputs)
  }

  def argmax(row: OnnxTensor): Int = {
    ???
  }

  def inplaceMatrixAddition(matrix: OnnxTensor, b: OnnxTensor): Unit = {
    ???
  }

  def inplaceVectorAddition(matrix: OnnxTensor, b: OnnxTensor): Unit = {
    ???
  }

  def mul(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def rows(matrix: OnnxTensor): Int = {
    ???
  }

  def cols(matrix: OnnxTensor): Int = {
    ???
  }

  def length(vector: OnnxTensor): Int = {
    ???
  }

  def t(matrix: OnnxTensor): OnnxTensor = {
    ???
  }

  def vertcat(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def zeros[T](rows: Int, cols: Int): OnnxTensor = {
    ???
  }

  def row(matrix: OnnxTensor, index: Int): OnnxTensor = {
    ???
  }

  def cat(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def toArray(vector: OnnxTensor): Array[Float] = {
    ???
  }

  def get(vector: OnnxTensor, index: Int): Float = {
    ???
  }

  def mkRowMatrix(values: Array[Array[Float]]): OnnxTensor = {
    ???
  }

  def mkVector(values: Array[Float]): OnnxTensor = {
    ???
  }
}
