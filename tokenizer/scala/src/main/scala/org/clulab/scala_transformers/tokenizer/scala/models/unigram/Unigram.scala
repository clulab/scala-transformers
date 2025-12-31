package org.clulab.scala_transformers.tokenizer.scala.models.unigram

import org.clulab.scala_transformers.tokenizer.scala.models.ModelChoice
import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.{JArray, JObject}

class Unigram(val unkId: Int, val vocab: Seq[Vocab]) extends ModelChoice {
  val typ = Unigram.typ
}

object Unigram extends JsonDeserializer[Unigram, JObject] {
  val typ = "Unigram"

  def deserialize(jObject: JObject): Unigram = {
    val typ = (jObject \ "type").extract[String]
    val unkId = (jObject \ "unk_id").extract[Int]
    val vocab = (jObject \ "vocab").extract[JArray].arr.map { jValue =>
      Vocab.deserialize(jValue.extract[JArray])
    }

    require(typ == this.typ)
    new Unigram(unkId, vocab)
  }
}
