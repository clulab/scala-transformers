package org.clulab.scala_transformers.encoder

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import breeze.linalg.DenseMatrix

import java.util.{HashMap => JHashMap}

class Encoder(val encoderEnvironment: OrtEnvironment, val encoderSession: OrtSession) {
  /**
    * Runs the inference using a transformer encoder over a batch of sentences
    *
    * @param batchInputIds First dimension is batch size (1 for a single sentence); second is sentence size
    * @return Hidden states for the whole batch. The matrix dimension: rows = sentence size; columns = hidden state size
    */
  def forward(batchInputIds: Array[Array[Long]]): Array[DenseMatrix[Float]] = {
    val inputs = new JHashMap[String, OnnxTensor]()
    inputs.put("token_ids", OnnxTensor.createTensor(encoderEnvironment, batchInputIds))

    val encoderOutput = encoderSession.run(inputs).get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val outputs = encoderOutput.map(BreezeUtils.mkRowMatrix(_))
    outputs
  }

  /** 
   * Runs inference over a single sentence 
   * @param inputIds Array of token ids for this sentence
   * @return Hidden states for this sentence. The matrix dimension: rows = sentence size; columns = hidden state size 
   */
  def forward(inputIds: Array[Long]): DenseMatrix[Float] = {
    val batchInputIds = Array(inputIds)
    forward(batchInputIds).head
  }
}

object Encoder {
  val ortEnvironment = OrtEnvironment.getEnvironment

  def apply(onnxModelFile: String): Encoder = {
    val ortSession = ortEnvironment.createSession(onnxModelFile, new OrtSession.SessionOptions)
    new Encoder(ortEnvironment, ortSession)
  }
}
