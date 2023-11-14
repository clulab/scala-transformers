package org.clulab.scala_transformers.encoder.math

import org.clulab.transformers.test.Test

class BreezeMathTest extends Test {

  def mkRowVector(values: Array[Float]): BreezeMath.MathRowVector = {
    BreezeMath.mkColVector(values).t
  }

  behavior of "BreezeMath"

  it should "argmax" in {
    val values = Array(1f, 3f, 2f)
    val vector = mkRowVector(values)
    val expectedResult = 1
    val actualResult = BreezeMath.argmax(vector)

    actualResult should be (expectedResult)
  }

  it should "inplaceMatrixAddition2" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkMatrixFromRows(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkColVector(vectorValues)
    val expectedResult = Array(
      Array(2f, 4f, 6f),
      Array(3f, 6f, 9f)
    )
    val actualResult = matrix

    BreezeMath.inplaceMatrixAddition(matrix, vector)
    expectedResult.zipWithIndex.foreach { case (expecteedValues, rowIndex) =>
      expecteedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult(rowIndex, colIndex) should be (expectedValue)
      }
    }
  }

  it should "inplaceMatrixAddition3" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkMatrixFromRows(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = mkRowVector(vectorValues)
    val expectedResult = Array(
      Array(1f, 2f, 3f),
      Array(3f, 6f, 9f)
    )
    val actualResult = matrix

    BreezeMath.inplaceMatrixAddition(matrix, 1, vector)
    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult(rowIndex, colIndex) should be(expectedValue)
      }
    }
  }

  it should "mul" in {
    val leftMatrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val leftMatrix = BreezeMath.mkMatrixFromRows(leftMatrixValues)
    val rightMatrixValues = Array(
      Array(1f, 2f),
      Array(3f, 2f),
      Array(4f, 6f)
    )
    val rightMatrix = BreezeMath.mkMatrixFromRows(rightMatrixValues)
    val expectedResult = Array(
      Array(19f, 24f),
      Array(38f, 48f)
    )
    val actualResult = BreezeMath.mul(leftMatrix, rightMatrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult(rowIndex, colIndex) should be(expectedValue)
      }
    }
  }

  it should "rows" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkMatrixFromRows(matrixValues)
    val expectedResult = 2
    val actualResult = BreezeMath.rows(matrix)

    actualResult should be (expectedResult)
  }

  it should "cols" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkMatrixFromRows(matrixValues)
    val expectedResult = 3
    val actualResult = BreezeMath.cols(matrix)

    actualResult should be(expectedResult)
  }

  it should "length" in {
    val values = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkColVector(values)
    val expectedResult = 3
    val actualResult = BreezeMath.length(vector)

    actualResult should be(expectedResult)
  }

  it should "vertcat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = BreezeMath.mkColVector(leftVectorValues)
    val rightVector = BreezeMath.mkColVector(rightVectorValues)
    val expectedResult = Array(1f, 2f, 3f, 2f, 4f, 6f)
    val actualResult = BreezeMath.vertcat(leftVector, rightVector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be(expectedValue)
    }
  }

  it should "zeros" in {
    val matrixValues = Array(
      Array(0f, 0f, 0f),
      Array(0f, 0f, 0f)
    )
    val expectedResult = matrixValues
    val actualResult = BreezeMath.zeros(2, 3)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult(rowIndex, colIndex) should be (expectedValue)
      }
    }
  }

  it should "row" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkMatrixFromRows(matrixValues)
    val expectedResult = Array(2f, 4f, 6f)
    val actualResult = BreezeMath.row(matrix, 1)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be(expectedValue)
    }
  }

  it should "cat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = mkRowVector(leftVectorValues)
    val rightVector = mkRowVector(rightVectorValues)
    val expectedResult = Array(1f, 2f, 3f, 2f, 4f, 6f)
    val actualResult = BreezeMath.horcat(leftVector, rightVector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be(expectedValue)
    }
  }

  it should "toArray" in {
    val values = Array(1f, 2f, 3f)
    val vector = mkRowVector(values)
    val expectedResult = values
    val actualResult = BreezeMath.toArray(vector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be (expectedValue)
    }
  }

  it should "get" in {
    val values = Array(1f, 2f, 3f)
    val vector = mkRowVector(values)
    val expectedResult = 2f
    val actualResult = BreezeMath.get(vector, 1)

    actualResult should be (expectedResult)
  }

  it should "mkRowMatrix" in {
    val matrix = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val expectedResult = matrix
    val actualResult = BreezeMath.mkMatrixFromRows(matrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult(rowIndex, colIndex) should be (expectedValue)
      }
    }
  }

  it should "mkColMatrix" in {
    val matrix = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val expectedResult = Array(
      Array(1f, 2f),
      Array(2f, 4f),
      Array(3f, 6f)
    )
    val actualResult = BreezeMath.mkMatrixFromCols(matrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult(rowIndex, colIndex) should be (expectedValue)
      }
    }
  }

  it should "mkVector" in {
    val vectorValues = Array(1f, 2f, 3f)
    val expectedResult = vectorValues
    val actualResult = BreezeMath.mkColVector(vectorValues)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be (expectedValue)
    }
  }
}
