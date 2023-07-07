package org.clulab.scala_transformers.encoder

class LinearLayerLayout(val baseName: String) {

  def name: String = s"$baseName/name"

  def dual: String = s"$baseName/dual"

  def weights: String = s"$baseName/weights"

  def biases: String = s"$baseName/biases"

  def labels: String = s"$baseName/labels"
}
