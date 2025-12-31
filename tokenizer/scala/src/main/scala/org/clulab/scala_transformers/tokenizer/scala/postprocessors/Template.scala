package org.clulab.scala_transformers.tokenizer.scala.postprocessors

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.{JArray, JObject}

class Template(val single: Seq[SingleProcessorsChoice], val pair: Seq[PairProcessorsChoice], val specialTokens: Map[String, SpecialTokens]) extends PostProcessorChoice{
  val typ = Template.typ
}

object Template extends JsonDeserializer[PostProcessorChoice, JObject] {
  val typ = "TemplateProcessing"

  def deserialize(jObject: JObject): PostProcessorChoice = {
    val typ = (jObject \ "type").extract[String]
    val single = (jObject \ "single").extract[JArray].arr.map { jValue =>
      Single.deserialize(jValue.extract[JObject])
    }
    val pair = (jObject \ "pair").extract[JArray].arr.map { jValue =>
      Pair.deserialize(jValue.extract[JObject])
    }
    val specialTokens = (jObject \ "special_tokens").extract[JObject].obj.map { jField =>
      val key = jField._1
      val jObject = jField._2.extract[JObject]
      val specialTokens = SpecialTokens.deserialize(jObject)

      key -> specialTokens
    }.toMap

    require(typ == this.typ)
    new Template(single, pair, specialTokens)
  }
}
