package org.clulab.scala_transformers.tokenizer.scala.normalizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

trait NormalizerChoice

object NormalizerChoice extends JsonDeserializer[NormalizerChoice, JObject] {

  def deserialize(jObject: JObject): NormalizerChoice = {
    val typ = (jObject \ "type").extract[String]

    typ match {
      case Sequence.typ => Sequence.deserialize(jObject)
      case _ => throw mkTypeError(this, typ)
    }
  }
}
