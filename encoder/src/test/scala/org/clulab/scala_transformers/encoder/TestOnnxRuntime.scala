package org.clulab.scala_transformers.encoder

import ai.onnxruntime.OrtEnvironment
import org.clulab.transformers.test.Test

class TestOnnxRuntime extends Test {
  behavior of "OnnxRuntime"

  it should "load" in {
    OrtEnvironment.getEnvironment
  }
}
