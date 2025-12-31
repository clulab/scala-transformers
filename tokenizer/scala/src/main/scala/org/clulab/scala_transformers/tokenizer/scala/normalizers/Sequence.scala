package org.clulab.scala_transformers.tokenizer.scala.normalizers

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.{JArray, JObject}

class Sequence(val normalizers: Seq[NormalizersChoice]) extends NormalizerChoice {
  val typ = Sequence.typ
}

object Sequence extends JsonDeserializer[NormalizerChoice, JObject] {
  val typ = "Sequence"

  def deserialize(jObject: JObject): NormalizerChoice = {
    val typ = (jObject \ "type").extract[String]
    val normalizers = (jObject \ "normalizers").extract[JArray].arr.map { jValue =>
      val jObject = jValue.extract[JObject]
      val typ = (jObject \ "type").extract[String]

      typ match {
        case Strip.typ => Strip.deserialize(jObject)
        case Replace.typ => Replace.deserialize(jObject)
        case Precompiled.typ => Precompiled.deserialize(jObject)
        case _ => throw mkTypeError(this, typ)
      }
    }

    require(typ == this.typ)
    new Sequence(normalizers)
  }
}
