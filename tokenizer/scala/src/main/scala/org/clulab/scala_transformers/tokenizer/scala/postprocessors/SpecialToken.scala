package org.clulab.scala_transformers.tokenizer.scala.postprocessors

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

class SpecialToken(val id: String, val typeId: Int) extends SingleProcessorsChoice with PairProcessorsChoice

object SpecialToken extends JsonDeserializer[SpecialToken, JObject] {
  def key = "SpecialToken"

  def deserialize(jObject: JObject): SpecialToken = {
    val id = (jObject \ "id").extract[String]
    val typeId = (jObject \ "type_id").extract[Int]

    new SpecialToken(id, typeId)
  }
}
