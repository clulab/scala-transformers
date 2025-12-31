package org.clulab.scala_transformers.tokenizer.scala.normalizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

class Precompiled(val precompiledCharsmap: String) extends NormalizersChoice {
  val typ = Precompiled.typ
}

object Precompiled extends JsonDeserializer[Precompiled, JObject] {
  val typ = "Precompiled"

  def deserialize(jObject: JObject): Precompiled = {
    val typ = (jObject \ "type").extract[String]
    val precompiledCharsmap = (jObject \ "precompiled_charsmap").extract[String]

    require(typ == this.typ)
    new Precompiled(precompiledCharsmap)
  }
}
