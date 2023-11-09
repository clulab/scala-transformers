package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.{OnnxTensor, OrtEnvironment}
import ai.onnxruntime.OrtSession.Result

object OnnxMath {
  type MathMatrix = OnnxTensor
  type MathVector = OnnxTensor
  type MathRowVector = OnnxTensor

  val ortEnvironment = OrtEnvironment.getEnvironment

  // This is 1 x 72 x 768
  // Turn it into an Array[OnnxTensor] with the tne tensor at 72 x 768
  def fromResult(result: Result): Array[OnnxTensor] = {
    val outputs = result.get("sequence_output").asInstanceOf[OnnxTensor]
//    val ans = outputs.getValue(0)
//
//    Array(ans)
    ???
  }

  def argmax(row: OnnxTensor): Int = {
    ???
  }

  def inplaceMatrixAddition(matrix: OnnxTensor, vector: OnnxTensor): Unit = {
    ???
  }

  def inplaceMatrixAddition(matrix: OnnxTensor, rowIndex: Int, vector: OnnxTensor): Unit = {
    ???
  }

  def mul(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def rows(matrix: OnnxTensor): Int = {
    matrix.getInfo.getShape()(0).toInt
  }

  def cols(matrix: OnnxTensor): Int = {
    matrix.getInfo.getShape()(1).toInt
  }

  def length(vector: OnnxTensor): Int = {
    vector.getInfo.getShape()(1).toInt
  }

  def t(matrix: OnnxTensor): OnnxTensor = {
    ???
  }

  def vertcat(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def zeros(rows: Int, cols: Int): OnnxTensor = {
    OnnxTensor.createTensor(ortEnvironment, Array.fill[Float](rows, cols)(0))
  }

  def row(matrix: OnnxTensor, index: Int): OnnxTensor = {
    ???
  }

  def cat(left: OnnxTensor, right: OnnxTensor): OnnxTensor = {
    ???
  }

  def toArray(vector: OnnxTensor): Array[Float] = {
    vector.getFloatBuffer.array
  }

  def get(vector: OnnxTensor, index: Int): Float = {
//    vector.getValue(index).asInstanceOf[Float]
    ???
  }

  def mkRowMatrix(values: Array[Array[Float]]): OnnxTensor = {
    OnnxTensor.createTensor(ortEnvironment, values)
  }

  def mkVector(values: Array[Float]): OnnxTensor = {
    OnnxTensor.createTensor(ortEnvironment, values)
  }
}
