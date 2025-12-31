package org.clulab.scala_transformers.tokenizer.scala.decoders

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

class Metaspace(val replacement: String, val addPrefixSpace: Boolean) extends Decoder {
  val typ = Metaspace.typ
}

object Metaspace extends JsonDeserializer[Metaspace, JObject] {
  val typ = "Metaspace"

  def deserialize(jObject: JObject): Metaspace = {
    val typ = (jObject \ "type").extract[String]
    val replacement = (jObject \ "replacement").extract[String]
    val addPrefixSpace = (jObject \ "add_prefix_space").extract[Boolean]

    require(typ == this.typ)
    new Metaspace(replacement, addPrefixSpace)
  }
}
