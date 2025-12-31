package org.clulab.scala_transformers.tokenizer.scala.pretokenizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.{JArray, JObject}

class Sequence(val pretokenizers: Seq[PreTokenizersChoice]) extends PreTokenizerChoice {
  val typ = Sequence.typ
}

object Sequence extends JsonDeserializer[Sequence, JObject] {
  val typ = "Sequence"

  def deserialize(jObject: JObject): Sequence = {
    val typ = (jObject \ "type").extract[String]
    val pretokenizers = (jObject \ "pretokenizers").extract[JArray].arr.map { jValue =>
      val jObject = jValue.extract[JObject]
      val typ = (jObject \ "type").extract[String]

      typ match {
        case Metaspace.typ => Metaspace.deserialize(jObject)
        case _ => throw mkTypeError(this, typ)
      }
    }
    require(typ == this.typ)
    new Sequence(pretokenizers)
  }
}
