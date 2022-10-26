package org.clulab.scala_transformers.encoder

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}

object LoadExample extends App {
  val words = Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".")
  val tokenizer = ScalaJniTokenizer("bert-base-cased")
  val tokenization = tokenizer.tokenize(words)

  val l = tokenization.tokens.length
  // we can add paddings here like ++ Array.fill(padded_length)(0) before grouped
  // I use group here to reshape the 1-D array to 2-D, we do not need it if we have batched inputs.
  val input_ids = tokenization.tokenIds.map(x => x.toLong).grouped(l).toArray
  for(i <- 0 until input_ids.length) {
    for(j <- 0 until input_ids(i).length) {
      print(input_ids(i)(j) + " ")
    }
    println()
  }

  val ortEnvironment = OrtEnvironment.getEnvironment
  val modelpath1 = "/Users/msurdeanu/github/scala-transformers/encoder.onnx"
  val session1 = ortEnvironment.createSession(modelpath1, new OrtSession.SessionOptions)
  val inputs = new java.util.HashMap[String, OnnxTensor]()
  inputs.put("token_ids", OnnxTensor.createTensor(ortEnvironment, input_ids))
  val bert_output = session1.run(inputs).get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]

  println(bert_output.length)
  for (o<-bert_output){
    println(o.length)
    for (o2<-o){
      println(o2.length)
      println(o2.mkString(","))
    }
  }
}