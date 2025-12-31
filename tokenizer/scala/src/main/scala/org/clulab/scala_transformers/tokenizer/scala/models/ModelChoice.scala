package org.clulab.scala_transformers.tokenizer.scala.models

import org.clulab.scala_transformers.tokenizer.scala.models.unigram.Unigram
import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

trait ModelChoice

object ModelChoice extends JsonDeserializer[ModelChoice, JObject] {

  def deserialize(jObject: JObject): ModelChoice = {
    val typ = (jObject \ "type").extract[String]

    typ match {
      case Unigram.typ => Unigram.deserialize(jObject)
      case _ => throw mkTypeError(this, typ)
    }
  }
}
