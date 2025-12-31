package org.clulab.scala_transformers.tokenizer.scala.decoders

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

trait Decoder

object Decoder extends JsonDeserializer[Decoder, JObject] {

  def deserialize(jObject: JObject): Decoder = {
    val typ = (jObject \ "type").extract[String]

    typ match {
      case Metaspace.typ => Metaspace.deserialize(jObject)
      case _ => throw mkTypeError(this, typ)
    }
  }
}
