package org.clulab.scala_transformers.tokenizer.scala.models.unigram

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JArray

class Vocab(val text: String, val score: Double)

object Vocab extends JsonDeserializer[Vocab, JArray] {

  def deserialize(jArray: JArray): Vocab = {
    val arr = jArray.arr

    assert(arr.length == 2)

    val text = arr(0).extract[String]
    val score = arr(1).extract[Double]

    new Vocab(text, score)
  }
}
