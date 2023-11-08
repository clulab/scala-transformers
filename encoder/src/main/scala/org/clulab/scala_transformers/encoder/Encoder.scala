package org.clulab.scala_transformers.encoder

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import org.clulab.scala_transformers.encoder.math.Math.{MathMatrix, Mathematician}

import java.io.DataInputStream
import java.util.{HashMap => JHashMap}

class Encoder(val encoderEnvironment: OrtEnvironment, val encoderSession: OrtSession) {
  /**
    * Runs the inference using a transformer encoder over a batch of sentences
    *
    * @param batchInputIds First dimension is batch size (1 for a single sentence); second is sentence size
    * @return Hidden states for the whole batch. The matrix dimension: rows = sentence size; columns = hidden state size
    */
  def forward(batchInputIds: Array[Array[Long]]): Array[MathMatrix] = {
    val inputs = new JHashMap[String, OnnxTensor]()
    inputs.put("token_ids", OnnxTensor.createTensor(encoderEnvironment, batchInputIds))

    val result: OrtSession.Result = encoderSession.run(inputs)
    val outputs = Mathematician.fromResult(result)
    outputs
  }

  /** 
   * Runs inference over a single sentence 
   * @param inputIds Array of token ids for this sentence
   * @return Hidden states for this sentence. The matrix dimension: rows = sentence size; columns = hidden state size 
   */
  def forward(inputIds: Array[Long]): MathMatrix = {
    val batchInputIds = Array(inputIds)
    forward(batchInputIds).head
  }
}

object Encoder {
  val ortEnvironment = OrtEnvironment.getEnvironment

  protected def fromSession(ortSession: OrtSession): Encoder =
      new Encoder(ortEnvironment, ortSession)

  protected def ortSessionFromFile(fileName: String): OrtSession =
      ortEnvironment.createSession(fileName, new OrtSession.SessionOptions)

  protected def ortSessionFromResource(resourceName: String): OrtSession = {
    val connection = getClass.getResource(resourceName).openConnection
    val contentLength = connection.getContentLength
    val bytes = new Array[Byte](contentLength)
    val inputStream = getClass.getResourceAsStream(resourceName)
    val dataInputStream = new DataInputStream(inputStream)

    try {
      dataInputStream.readFully(bytes)
    }
    finally {
      dataInputStream.close()
    }
    ortEnvironment.createSession(bytes, new OrtSession.SessionOptions)
  }

  def fromFile(onnxModelFile: String): Encoder =
      fromSession(ortSessionFromFile(onnxModelFile))

  def fromResource(resourceName: String): Encoder =
      fromSession(ortSessionFromResource(resourceName))
}
