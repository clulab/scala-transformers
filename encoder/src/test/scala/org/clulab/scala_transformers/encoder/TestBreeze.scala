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

  it should "add vectors to matrix rows" in {
    val m = Array(1, 2, 3, 4, 5, 6)
    val dm = DenseMatrix.create(rows = 3, cols = 2, m)
    println(dm)

    val dv = DenseVector[Int](1, 1)

    dm(0, ::) :+= dv.t
    println(dm)

    dm(0, 0) should be (2)
    dm(0, 1) should be (5)
  }

  it should "add vectors to matrix columns" in {
    val m = Array(1, 2, 3, 4, 5, 6)
    val dm = DenseMatrix.create(rows = 3, cols = 2, m)
    println(dm)

    val dv = DenseVector[Int](1, 1, 1)

    dm(::, 0) :+= dv
    println(dm)

    dm(0, 0) should be (2)
    dm(1, 0) should be (3)
    dm(2, 0) should be (4)
  }

  it should "broadcast correctly by rows" in {
    val m = Array(1, 2, 3, 4, 5, 6)
    val dm = DenseMatrix.create(rows = 3, cols = 2, m)
    println(dm)

    val dv = DenseVector[Int](1, 1)

    // add dv to each row
    dm(*, ::) :+= dv

    dm(0, 0) should be (2)
    dm(0, 1) should be (5)
    dm(1, 0) should be (3)
    dm(1, 1) should be (6)
    dm(2, 0) should be (4)
    dm(2, 1) should be (7)
  }

  it should "broadcast correctly by columns" in {
    val m = Array(1, 2, 3, 4, 5, 6)
    val dm = DenseMatrix.create(rows = 3, cols = 2, m)
    println(dm)

    val dv = DenseVector[Int](1, 1, 1)

    // add dv to each row
    dm(::, *) :+= dv

    dm(0, 0) should be (2)
    dm(0, 1) should be (5)
    dm(1, 0) should be (3)
    dm(1, 1) should be (6)
    dm(2, 0) should be (4)
    dm(2, 1) should be (7)
  }
}
