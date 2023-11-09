package org.clulab.scala_transformers.encoder.math

import ai.onnxruntime.OrtSession.Result

trait Math {
  type MathRowMatrix
  type MathValue
  type MathColVector
  type MathRowVector

  def fromResult(result: Result): Array[MathRowMatrix]
  def argmax(rowVector: MathRowVector): Int
  def inplaceMatrixAddition(matrix: MathRowMatrix, colVector: MathColVector): Unit
  def inplaceMatrixAddition(matrix: MathRowMatrix, rowIndex: Int, rowVector: MathRowVector): Unit
  def mul(leftMatrix: MathRowMatrix, rightMatrix: MathRowMatrix): MathRowMatrix
  def rows(matrix: MathRowMatrix): Int
  def cols(matrix: MathRowMatrix): Int
  def length(colVector: MathColVector): Int
  def t(matrix: MathRowMatrix): MathRowMatrix
  def vertcat(leftColVector: MathColVector, rightColVector: MathColVector): MathColVector
  def zeros(rows: Int, cols: Int): MathRowMatrix
  def row(matrix: MathRowMatrix, index: Int): MathRowVector
  def cat(leftRowVector: MathRowVector, rightRowVector: MathRowVector): MathRowVector
  def toArray(rowVector: MathRowVector): Array[MathValue]
  def get(rowVector: MathRowVector, index: Int): MathValue
  def mkRowMatrix(values: Array[Array[MathValue]]): MathRowMatrix
  def mkVector(values: Array[MathValue]): MathColVector
}
