package org.clulab.scala_transformers.encoder

import breeze.linalg._

object BreezeExample extends App {
  val a = Array(1, 2, 3)
  val dv = Transpose(DenseVector(a))
  println(dv)

  val m = Array(1, 2, 3, 4, 5, 6)
  val dm = DenseMatrix.create(rows = 3, cols = 2, m)
  println(dm)

  val bias = DenseVector(1, 1)

  val prod = dv * dm + Transpose(bias)
  print(prod)
}
