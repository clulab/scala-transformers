package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.encoder.math.Mathematics.MathValue

trait NonLinearity {
  def compute(input: MathValue): MathValue
}

object ReLU extends NonLinearity {
  override def compute(input: MathValue): MathValue = {
    scala.math.max(0, input)
  }
}
