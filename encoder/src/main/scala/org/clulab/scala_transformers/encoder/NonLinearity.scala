package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.encoder.math.EjmlMath.MathValue

import java.lang

trait NonLinearity {
  def compute(input: MathValue): MathValue
}

class ReLU extends NonLinearity {
  override def compute(input: MathValue): MathValue = {
    lang.Float.max(0, input)
  }
}
