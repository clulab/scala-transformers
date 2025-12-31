package org.clulab.scala_transformers.tokenizer.scala.postprocessors

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

class Sequence(val id: String, val typeId: Int) extends SingleProcessorsChoice with PairProcessorsChoice

object Sequence extends JsonDeserializer[Sequence, JObject] {
  val key = "Sequence"

  def deserialize(jObject: JObject): Sequence = {
    val id = (jObject \ "id").extract[String]
    val typeId = (jObject \ "type_id").extract[Int]

    new Sequence(id, typeId)
  }
}
