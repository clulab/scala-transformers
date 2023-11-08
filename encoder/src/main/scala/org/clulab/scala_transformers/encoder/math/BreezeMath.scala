package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result
import breeze.linalg.`*`
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.{argmax => BreezeArgmax}
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
}
