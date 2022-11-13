package org.clulab.scala_transformers.encoder

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}

import breeze.linalg._

class Encoder(val encoderEnvironment: OrtEnvironment, val encoderSession: OrtSession) {
  /**
    * Runs the inference using a transformer encoder over a batch of sentences
    *
    * @param batchInputIds First dimension is batch size (1 for a single sentence); second is sentence size
    * @return Hidden states for the whole batch. The matrix dimension: rows = sentence size; columns = hidden state size
    */
  def forward(batchInputIds: Array[Array[Long]]): Array[DenseMatrix[Float]] = {
    val inputs = new java.util.HashMap[String, OnnxTensor]()
    inputs.put("token_ids", OnnxTensor.createTensor(encoderEnvironment, batchInputIds))
    val encoderOutput = encoderSession.run(inputs).get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]

    val outputs = new Array[DenseMatrix[Float]](encoderOutput.length)
    for(i <- encoderOutput.indices) {
      outputs(i) = BreezeUtils.mkRowMatrix(encoderOutput(i))
    }

    outputs
  }

  /** 
   * Runs inference over a single sentence 
   * @param inputIds Array of token ids for this sentence
   * @return Hidden states for this sentence. The matrix dimension: rows = sentence size; columns = hidden state size 
   */
  def forward(inputIds: Array[Long]): DenseMatrix[Float] = {
    val batchInputIds = new Array[Array[Long]](1)
    batchInputIds(0) = inputIds
    forward(batchInputIds)(0)
  }
}

object Encoder {
  def apply(onnxModelFile: String): Encoder = {
    val ortEnvironment = OrtEnvironment.getEnvironment
    val ortSession = ortEnvironment.createSession(onnxModelFile, new OrtSession.SessionOptions)
    new Encoder(ortEnvironment, ortSession)
  }
}
