package org.clulab.scala_transformers.tokenizer.scala.postprocessors

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

trait PostProcessorChoice

object PostProcessorChoice extends JsonDeserializer[PostProcessorChoice, JObject] {

  def deserialize(jObject: JObject): PostProcessorChoice = {
    val typ = (jObject \ "type").extract[String]

    typ match {
      case Template.typ => Template.deserialize(jObject)
      case _ => throw mkTypeError(this, typ)
    }
  }
}
