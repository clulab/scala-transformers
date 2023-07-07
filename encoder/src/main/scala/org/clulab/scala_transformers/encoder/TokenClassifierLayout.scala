package org.clulab.scala_transformers.encoder

class TokenClassifierLayout(val baseName: String) {

  def onnx: String = s"$baseName/encoder.onnx"

  def name: String = s"$baseName/encoder.name"

  def tasks: String = s"$baseName/tasks"

  def task(index: Int): String = s"${tasks}/$index"

  def linearLayerLayout(index: Int): LinearLayerLayout = new LinearLayerLayout(task(index))

  def addPrefixSpace: Boolean = baseName.toLowerCase().contains("roberta")
}
