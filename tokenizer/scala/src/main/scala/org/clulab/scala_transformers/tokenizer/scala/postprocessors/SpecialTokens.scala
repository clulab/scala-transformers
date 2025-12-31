package org.clulab.scala_transformers.tokenizer.scala.postprocessors

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.{JArray, JObject}

class SpecialTokens(id: String, ids: Seq[Int], tokens: Seq[String])

object SpecialTokens extends JsonDeserializer[SpecialTokens, JObject] {

  def deserialize(jObject: JObject): SpecialTokens = {
    val id = (jObject \ "id").extract[String]
    val ids = (jObject \ "ids").extract[JArray].arr.map { jValue =>
      jValue.extract[Int]
    }
    val tokens = (jObject \ "tokens").extract[JArray].arr.map { jValue =>
      jValue.extract[String]
    }

    new SpecialTokens(id, ids, tokens)
  }
}
