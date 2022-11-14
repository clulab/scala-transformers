package org.clulab.scala_transformers.encoder

import breeze.linalg.DenseMatrix

import scala.reflect.ClassTag

object BreezeUtils {
  /** 
   * Constructs a dense matrix by rows 
   * 
   * @param inputs Input values; first dimension are rows; second are columns
   */
  def mkRowMatrix[T: ClassTag](inputs: Array[Array[T]]): DenseMatrix[T] = {
    val rows = inputs.length
    val cols = inputs.head.length
    val dm = new DenseMatrix[T](rows, cols)

    for (i <- 0 until rows)
      for (j <- 0 until cols)
        dm(i, j) = inputs(i)(j)
    dm
  }

  def arrayConcat[T: ClassTag](arrays: Array[Array[T]]): Array[T] = {
    val indivLen = arrays.head.length
    val overallLen = arrays.length * indivLen
    val output = new Array[T](overallLen)

    for (i <- arrays.indices)
      System.arraycopy(arrays(i), 0, output, i * indivLen, indivLen)
    output
  }
}
