package org.clulab.scala_transformers.encoder.math

import org.clulab.transformers.test.Test

class EjmlMathTest extends Test {

  behavior of "Math"

  it should "argmax" in {
    val values = Array(1f, 3f, 2f)
    val vector = EjmlMath.t(EjmlMath.mkVector(values))
    val expectedResult = 1
    val actualResult = EjmlMath.argmax(vector)

    actualResult should be (expectedResult)
  }

  it should "inplaceMatrixAddition2" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = EjmlMath.mkRowMatrix(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = EjmlMath.mkVector(vectorValues)
    val expectedResult = Array(
      Array(2f, 4f, 6f),
      Array(3f, 6f, 9f)
    )
    val actualResult = matrix

    EjmlMath.inplaceMatrixAddition(matrix, vector)
    expectedResult.zipWithIndex.foreach { case (expecteedValues, rowIndex) =>
      expecteedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.get(rowIndex, colIndex) should be (expectedValue)
      }
    }
  }

  it should "inplaceMatrixAddition3" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = EjmlMath.mkRowMatrix(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = EjmlMath.t(EjmlMath.mkVector(vectorValues))
    val expectedResult = Array(
      Array(1f, 2f, 3f),
      Array(3f, 6f, 9f)
    )
    val actualResult = matrix

    EjmlMath.inplaceMatrixAddition(matrix, 1, vector)
    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.get(rowIndex, colIndex) should be(expectedValue)
      }
    }
  }

  it should "mul" in {
    val leftMatrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val leftMatrix = EjmlMath.mkRowMatrix(leftMatrixValues)
    val rightMatrixValues = Array(
      Array(1f, 2f),
      Array(3f, 2f),
      Array(4f, 6f)
    )
    val rightMatrix = EjmlMath.mkRowMatrix(rightMatrixValues)
    val expectedResult = Array(
      Array(19f, 24f),
      Array(38f, 48f)
    )
    val actualResult = EjmlMath.mul(leftMatrix, rightMatrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.get(rowIndex, colIndex) should be(expectedValue)
      }
    }
  }

  it should "rows" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = EjmlMath.mkRowMatrix(matrixValues)
    val expectedResult = 2
    val actualResult = EjmlMath.rows(matrix)

    actualResult should be (expectedResult)
  }

  it should "cols" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = EjmlMath.mkRowMatrix(matrixValues)
    val expectedResult = 3
    val actualResult = EjmlMath.cols(matrix)

    actualResult should be(expectedResult)
  }

  it should "length" in {
    val values = Array(1f, 2f, 3f)
    val vector = EjmlMath.mkVector(values)
    val expectedResult = 3
    val actualResult = EjmlMath.length(vector)

    actualResult should be(expectedResult)
  }

  it should "t" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = EjmlMath.mkRowMatrix(matrixValues)
    val expectedResult = Array(
      Array(1f, 2f),
      Array(2f, 4f),
      Array(3f, 6f)
    )
    val actualResult = EjmlMath.t(matrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.get(rowIndex, colIndex) should be(expectedValue)
      }
    }
  }

  it should "vertcat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = EjmlMath.mkVector(leftVectorValues)
    val rightVector = EjmlMath.mkVector(rightVectorValues)
    val expectedResult = Array(1f, 2f, 3f, 2f, 4f, 6f)
    val actualResult = EjmlMath.vertcat(leftVector, rightVector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.get(index) should be(expectedValue)
    }
  }

  it should "zeros" in {
    val matrixValues = Array(
      Array(0f, 0f, 0f),
      Array(0f, 0f, 0f)
    )
    val expectedResult = matrixValues
    val actualResult = EjmlMath.zeros(2, 3)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.get(rowIndex, colIndex) should be (expectedValue)
      }
    }
  }

  it should "row" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = EjmlMath.mkRowMatrix(matrixValues)
    val expectedResult = Array(2f, 4f, 6f)
    val actualResult = EjmlMath.row(matrix, 1)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.get(index) should be(expectedValue)
    }
  }

  it should "cat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = EjmlMath.t(EjmlMath.mkVector(leftVectorValues))
    val rightVector = EjmlMath.t(EjmlMath.mkVector(rightVectorValues))
    val expectedResult = Array(1f, 2f, 3f, 2f, 4f, 6f)
    val actualResult = EjmlMath.cat(leftVector, rightVector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.get(index) should be(expectedValue)
    }
  }

  it should "toArray" in {
    val values = Array(1f, 2f, 3f)
    val vector = EjmlMath.t(EjmlMath.mkVector(values))
    val expectedResult = values
    val actualResult = EjmlMath.toArray(vector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be (expectedValue)
    }
  }

  it should "get" in {
    val values = Array(1f, 2f, 3f)
    val vector = EjmlMath.t(EjmlMath.mkVector(values))
    val expectedResult = 2f
    val actualResult = EjmlMath.get(vector, 1)

    actualResult should be (expectedResult)
  }

  it should "mkRowMatrix" in {
    val values = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val expectedResult = values
    val actualResult = EjmlMath.mkRowMatrix(values)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.get(rowIndex, colIndex) should be (expectedValue)
      }
    }
  }

  it should "mkVector" in {
    val values = Array(1f, 2f, 3f)
    val expectedResult = values
    val actualResult = EjmlMath.mkVector(expectedResult)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.get(index) should be (expectedValue)
    }
  }
}