package org.clulab.scala_transformers.tokenizer.scala.postprocessors

import org.clulab.scala_transformers.tokenizer.scala.postprocessors.Single.mkKeyError
import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

object Pair extends JsonDeserializer[PairProcessorsChoice, JObject] {

  def deserialize(jObject: JObject): PairProcessorsChoice = {
    (jObject \ SpecialToken.key).extractOpt[JObject].map { jObject =>
      SpecialToken.deserialize(jObject)
    }
    .orElse {
      (jObject \ Sequence.key).extractOpt[JObject].map { jObject =>
        Sequence.deserialize(jObject)
      }
    }
    .getOrElse {
      throw mkKeyError(this, SpecialToken.key, Sequence.key)
    }
  }
}
