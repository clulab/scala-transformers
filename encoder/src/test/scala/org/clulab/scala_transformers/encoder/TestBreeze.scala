package org.clulab.scala_transformers.encoder

import org.clulab.transformers.test.Test

import breeze.linalg._

class TestBreeze extends Test {
  behavior of "Breeze"

  it should "perform multiplications and additions correctly" in {
    val a = Array(1, 2, 3)
    val dv = DenseMatrix.create(rows = 1, cols = 3, a)
    println(dv)

    val m = Array(1, 2, 3, 4, 5, 6)
    val dm = DenseMatrix.create(rows = 3, cols = 2, m)
    println(dm)

    val bias = DenseVector(1, 1)

    // simulates a linear layer
    val prod = dv * dm + bias.t
    println(prod)

    prod.rows should be (1)
    prod.cols should be (2)
    prod(0, 0) should be (15)
    prod(0, 1) should be (33)
  }
}
