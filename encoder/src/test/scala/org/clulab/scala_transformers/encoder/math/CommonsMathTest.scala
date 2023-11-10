package org.clulab.scala_transformers.encoder.math

import org.apache.commons.math3.linear.ArrayRealVector
import org.clulab.scala_transformers.encoder.math.CommonsMath.{MathColVector, mkColVector}
import org.clulab.transformers.test.Test

class CommonsMathTest extends Test {

  def mkRowVector(values: Array[Float]): MathColVector = {
    mkColVector(values)
  }

  behavior of "Math"

  it should "argmax" in {
    val vectorValues = Array(1f, 3f, 2f)
    val vector = new ArrayRealVector(vectorValues.map(_.toDouble))
    val expectedResult = 1
    val actualResult = CommonsMath.argmax(vector)

    actualResult should be (expectedResult)
  }

  it should "inplaceMatrixAddition2" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = CommonsMath.mkMatrixFromRows(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = CommonsMath.mkColVector(vectorValues)
    val expectedResult = Array(
      Array(2f, 4f, 6f),
      Array(3f, 6f, 9f)
    )
    val actualResult = matrix

    CommonsMath.inplaceMatrixAddition(matrix, vector)
    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.getEntry(rowIndex, colIndex).toFloat should be (expectedValue)
      }
    }
  }

  it should "inplaceMatrixAddition3" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = CommonsMath.mkMatrixFromRows(matrixValues)
    val vectorValues = Array(1f, 2f, 3f)
    val vector = mkRowVector(vectorValues)
    val expectedResult = Array(
      Array(1f, 2f, 3f),
      Array(3f, 6f, 9f)
    )
    val actualResult = matrix

    CommonsMath.inplaceMatrixAddition(matrix, 1, vector)
    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.getEntry(rowIndex, colIndex).toFloat should be(expectedValue)
      }
    }
  }

  it should "mul" in {
    val leftMatrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val leftMatrix = CommonsMath.mkMatrixFromRows(leftMatrixValues)
    val rightMatrixValues = Array(
      Array(1f, 2f),
      Array(3f, 2f),
      Array(4f, 6f)
    )
    val rightMatrix = CommonsMath.mkMatrixFromRows(rightMatrixValues)
    val expectedResult = Array(
      Array(19f, 24f),
      Array(38f, 48f)
    )
    val actualResult = CommonsMath.mul(leftMatrix, rightMatrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.getEntry(rowIndex, colIndex).toFloat should be(expectedValue)
      }
    }
  }

  it should "rows" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = CommonsMath.mkMatrixFromRows(matrixValues)
    val expectedResult = 2
    val actualResult = CommonsMath.rows(matrix)

    actualResult should be (expectedResult)
  }

  it should "cols" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = CommonsMath.mkMatrixFromRows(matrixValues)
    val expectedResult = 3
    val actualResult = CommonsMath.cols(matrix)

    actualResult should be(expectedResult)
  }

  it should "length" in {
    val vectorValues = Array(1f, 2f, 3f)
    val vector = CommonsMath.mkColVector(vectorValues)
    val expectedResult = 3
    val actualResult = CommonsMath.length(vector)

    actualResult should be(expectedResult)
  }

  it should "t" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = CommonsMath.mkMatrixFromRows(matrixValues)
    val expectedResult = Array(
      Array(1f, 2f),
      Array(2f, 4f),
      Array(3f, 6f)
    )
    val actualResult = CommonsMath.t(matrix)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.getEntry(rowIndex, colIndex).toFloat should be(expectedValue)
      }
    }
  }

  it should "vertcat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = CommonsMath.mkColVector(leftVectorValues)
    val rightVector = CommonsMath.mkColVector(rightVectorValues)
    val expectedResult = Array(1f, 2f, 3f, 2f, 4f, 6f)
    val actualResult = CommonsMath.vertcat(leftVector, rightVector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.getEntry(index).toFloat should be(expectedValue)
    }
  }

  it should "zeros" in {
    val matrixValues = Array(
      Array(0f, 0f, 0f),
      Array(0f, 0f, 0f)
    )
    val expectedResult = matrixValues
    val actualResult = CommonsMath.zeros(2, 3)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.getEntry(rowIndex, colIndex).toFloat should be (expectedValue)
      }
    }
  }

  it should "row" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val matrix = CommonsMath.mkMatrixFromRows(matrixValues)
    val expectedResult = Array(2f, 4f, 6f)
    val actualResult = CommonsMath.row(matrix, 1)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.getEntry(index).toFloat should be(expectedValue)
    }
  }

  it should "cat" in {
    val leftVectorValues = Array(1f, 2f, 3f)
    val rightVectorValues = Array(2f, 4f, 6f)
    val leftVector = mkRowVector(leftVectorValues)
    val rightVector = mkRowVector(rightVectorValues)
    val expectedResult = Array(1f, 2f, 3f, 2f, 4f, 6f)
    val actualResult = CommonsMath.horcat(leftVector, rightVector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.getEntry(index).toFloat should be(expectedValue)
    }
  }

  it should "toArray" in {
    val vectorValues = Array(1f, 2f, 3f)
    val vector = mkRowVector(vectorValues)
    val expectedResult = vectorValues
    val actualResult = CommonsMath.toArray(vector)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult(index) should be (expectedValue)
    }
  }

  it should "get" in {
    val vectorValues = Array(1f, 2f, 3f)
    val vector = mkRowVector(vectorValues)
    val expectedResult = 2f
    val actualResult = CommonsMath.get(vector, 1).toFloat

    actualResult should be (expectedResult)
  }

  it should "mkRowMatrix" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val expectedResult = matrixValues
    val actualResult = CommonsMath.mkMatrixFromRows(matrixValues)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.getEntry(rowIndex, colIndex).toFloat should be (expectedValue)
      }
    }
  }

  it should "mkColMatrix" in {
    val matrixValues = Array(
      Array(1f, 2f, 3f),
      Array(2f, 4f, 6f)
    )
    val expectedResult = Array(
      Array(1f, 2f),
      Array(2f, 4f),
      Array(3f, 6f)
    )
    val actualResult = CommonsMath.mkMatrixFromCols(matrixValues)

    expectedResult.zipWithIndex.foreach { case (expectedValues, rowIndex) =>
      expectedValues.zipWithIndex.foreach { case (expectedValue, colIndex) =>
        actualResult.getEntry(rowIndex, colIndex).toFloat should be(expectedValue)
      }
    }
  }

  it should "mkVector" in {
    val vectorValues = Array(1f, 2f, 3f)
    val expectedResult = vectorValues
    val actualResult = CommonsMath.mkColVector(expectedResult)

    expectedResult.zipWithIndex.foreach { case (expectedValue, index) =>
      actualResult.getEntry(index).toFloat should be (expectedValue)
    }
  }
}
