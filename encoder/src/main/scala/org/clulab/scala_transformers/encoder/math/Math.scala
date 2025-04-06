package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result

trait Math {
  type MathValue
  type MathRowMatrix
  type MathColVector
  type MathRowVector

  def fromResult(result: Result): Array[MathRowMatrix]
  def argmax(rowVector: MathRowVector): Int
  def inplaceMatrixAddition(matrix: MathRowMatrix, colVector: MathColVector): Unit
  def inplaceMatrixAddition(matrix: MathRowMatrix, rowIndex: Int, rowVector: MathRowVector): Unit
//  def rowVectorAddition(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector
  def map(matrix: MathRowMatrix, f: MathValue => MathValue): Unit
  def mul(leftMatrix: MathRowMatrix, rightMatrix: MathRowMatrix): MathRowMatrix
  def rows(matrix: MathRowMatrix): Int
  def cols(matrix: MathRowMatrix): Int
  def length(colVector: MathColVector): Int
  def vertcat(leftColVector: MathColVector, rightColVector: MathColVector): MathColVector
  def zeros(rows: Int, cols: Int): MathRowMatrix
  def row(matrix: MathRowMatrix, index: Int): MathRowVector
  def horcat(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector
  def toArray(rowVector: MathRowVector): Array[MathValue]
  def get(rowVector: MathRowVector, index: Int): MathValue
  def set(rowVector: MathRowVector, index: Int, value: MathValue): Unit
  def mkMatrixFromRows(values: Array[Array[MathValue]]): MathRowMatrix
  // For this, the array is specified in column-major order,
  // but it should be converted to the normal representation.
  def mkMatrixFromCols(values: Array[Array[MathValue]]): MathRowMatrix
  def mkColVector(values: Array[MathValue]): MathColVector
}
