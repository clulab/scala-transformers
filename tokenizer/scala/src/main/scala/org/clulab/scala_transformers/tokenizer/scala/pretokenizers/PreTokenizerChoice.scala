package org.clulab.scala_transformers.tokenizer.scala.pretokenizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

trait PreTokenizerChoice

object PreTokenizerChoice extends JsonDeserializer[PreTokenizerChoice, JObject] {

  def deserialize(jObject: JObject): PreTokenizerChoice = {
    val typ = (jObject \ "type").extract[String]

    typ match {
      case Sequence.typ => Sequence.deserialize(jObject)
      case _ => throw mkTypeError(this, typ)
    }
  }
}
