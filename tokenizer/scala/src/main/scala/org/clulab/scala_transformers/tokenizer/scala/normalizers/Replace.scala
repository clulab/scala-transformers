package org.clulab.scala_transformers.tokenizer.scala.normalizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

class Pattern(val regex: String)

object Pattern extends JsonDeserializer[Pattern, JObject] {

  def deserialize(jObject: JObject): Pattern = {
    val regex = (jObject \ "Regex").extract[String]

    new Pattern(regex)
  }
}

class Replace(val pattern: Pattern, val content: String) extends NormalizersChoice {
  val typ = Replace.typ
}

object Replace extends JsonDeserializer[Replace, JObject] {
  val typ = "Replace"

  def deserialize(jObject: JObject): Replace = {
    val typ = (jObject \ "type").extract[String]
    val pattern = Pattern.deserialize((jObject \ "pattern").extract[JObject])
    val content = (jObject \ "content").extract[String]

    require(typ == this.typ)
    new Replace(pattern, content)
  }
}
