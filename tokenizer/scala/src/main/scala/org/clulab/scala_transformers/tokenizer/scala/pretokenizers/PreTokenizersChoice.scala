package org.clulab.scala_transformers.tokenizer.scala.pretokenizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.{JArray, JObject}

trait PreTokenizersChoice

object PreTokenizersChoice extends JsonDeserializer[PreTokenizerChoice, JObject] {

  def deserialize(jObject: JObject): Sequence = {
    val pretokenizers = (jObject \ "pretokenizers").extract[JArray].arr.map { jValue =>
      val jObject = jValue.extract[JObject]
      val typ = (jObject \ "type").extract[String]

      typ match {
        case Metaspace.typ => Metaspace.deserialize(jObject)
        case _ => throw mkTypeError(this, typ)
      }
    }
    new Sequence(pretokenizers)
  }
}