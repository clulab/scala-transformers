package org.clulab.scala_transformers.encoder.math

import org.clulab.transformers.test.Test

class BreezeMathTest extends Test {

  behavior of "Math"

  it should "argmax" in {
    val values = Array(1f, 3f, 2f)
    val vector = BreezeMath.mkVector(values).t
    val expectedResult = 1
    val actualResult = BreezeMath.argmax(vector)

    actualResult should be (expectedResult)
  }

  it should "inplaceMatrixAddition2" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkRowMatrix(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkVector(vectorValues)
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
    val matrix = BreezeMath.mkRowMatrix(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkVector(vectorValues).t
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
    val leftMatrix = BreezeMath.mkRowMatrix(leftMatrixValues)
    val rightMatrixValues = Array(
      Array(1f, 2f),
      Array(3f, 2f),
      Array(4f, 6f)
    )
    val rightMatrix = BreezeMath.mkRowMatrix(rightMatrixValues)
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
    val matrix = BreezeMath.mkRowMatrix(matrixValues)
    val expectedResult = 2
    val actualResult = BreezeMath.rows(matrix)

    actualResult should be (expectedResult)
  }

  it should "cols" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkRowMatrix(matrixValues)
    val expectedResult = 3
    val actualResult = BreezeMath.cols(matrix)

    actualResult should be(expectedResult)
  }

  it should "length" in {
    val values = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkVector(values)
    val expectedResult = 3
    val actualResult = BreezeMath.length(vector)

    actualResult should be(expectedResult)
  }

  it should "t" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkRowMatrix(matrixValues)
    val expectedResult = Array(
      Array(1f, 2f),
      Array(2f, 4f),
      Array(3f, 6f)
    )
    val actualResult = BreezeMath.t(matrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult(rowIndex, colIndex) should be(expectedValue)
      }
    }
  }

  it should "vertcat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = BreezeMath.mkVector(leftVectorValues)
    val rightVector = BreezeMath.mkVector(rightVectorValues)
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
    val matrix = BreezeMath.mkRowMatrix(matrixValues)
    val expectedResult = Array(2f, 4f, 6f)
    val actualResult = BreezeMath.row(matrix, 1)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be(expectedValue)
    }
  }

  it should "cat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = BreezeMath.mkVector(leftVectorValues).t
    val rightVector = BreezeMath.mkVector(rightVectorValues).t
    val expectedResult = Array(1f, 2f, 3f, 2f, 4f, 6f)
    val actualResult = BreezeMath.cat(leftVector, rightVector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be(expectedValue)
    }
  }

  it should "toArray" in {
    val values = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkVector(values).t
    val expectedResult = values
    val actualResult = BreezeMath.toArray(vector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be (expectedValue)
    }
  }

  it should "get" in {
    val values = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkVector(values).t
    val expectedResult = 2f
    val actualResult = BreezeMath.get(vector, 1)

    actualResult should be (expectedResult)
  }

  it should "mkRowMatrix" in {
    val values = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = BreezeMath.mkRowMatrix(values)
    val expectedResult = "1.0  2.0  3.0  \n2.0  4.0  6.0  "
    val actualResult = matrix.toString

    actualResult should be (expectedResult)
    values.zipWithIndex.foreach { case (values, rowIndex) =>
      values.zipWithIndex.foreach { case (value, colIndex) =>
        matrix(rowIndex, colIndex) should be (value)
      }
    }
  }

  it should "mkVector" in {
    val values = Array(1f, 2f, 3f)
    val vector = BreezeMath.mkVector(values)
    val expectedResult = "DenseVector(1.0, 2.0, 3.0)"
    val actualResult = vector.toString

    actualResult should be (expectedResult)
    values.zipWithIndex.foreach { case (value, index) =>
      vector(index) should be (value)
    }
  }
}
