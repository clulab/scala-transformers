package org.clulab.scala_transformers.tokenizer.scala.normalizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

class Strip(val stripLeft: Boolean, val stripRight: Boolean) extends NormalizersChoice {
  val typ = Strip.typ
}

object Strip extends JsonDeserializer[Strip, JObject] {
  val typ = "Strip"

  def deserialize(jObject: JObject): Strip = {
    val typ = (jObject \ "type").extract[String]
    val stripLeft = (jObject \ "strip_left").extract[Boolean]
    val stripRight = (jObject \ "strip_right").extract[Boolean]

    require(typ == this.typ)
    new Strip(stripLeft, stripRight)
  }
}
