package org.clulab.scala_transformers.encoder

import breeze.linalg._ 
import BreezeUtils._

object BreezeExamples extends App {
  val m = mkRowMatrix[Float](Array(Array(1f, 2f), Array(3f, 4f)))
  println(m)
  
  println("Row 0: " + m(0, ::))

  // vertcat operates over vertical vectors, so we need to transpose the rows
  val cm = DenseVector.vertcat(m(0, ::).t, m(1, ::).t)
  println("Rows 0 and 1 concatenated: " + cm)

  val dm = DenseMatrix.zeros[Float](rows = m.rows, cols = 2 * m.cols)
  println("Initial dm:")
  println(dm)

  dm(0, ::) :+= cm.t
  dm(1, ::) :+= DenseVector.vertcat(m(1, ::).t, m(1, ::).t).t
  println("After changing dm:")
  println(dm)
}