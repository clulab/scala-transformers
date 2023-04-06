package org.clulab.scala_transformers.encoder

class ModelLayout(val baseName: String) {

  def getModel: String = s"$baseName/encoder.onnx"

  def getName: String = s"$baseName/encoder.name"

  def getTasks: String = s"$baseName/tasks"

  def getTask(index: Int): String = s"${getTasks}/$index"

  def getTaskName(index: Int): String = s"${getTask(index)}/name"

  def getTaskDual(index: Int): String = s"${getTask(index)}/dual"

  def addPrefixSpace: Boolean = baseName.toLowerCase().contains("roberta")
}
