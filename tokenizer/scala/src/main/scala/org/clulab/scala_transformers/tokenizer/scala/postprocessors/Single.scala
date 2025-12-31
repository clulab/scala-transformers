package org.clulab.scala_transformers.tokenizer.scala.postprocessors

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.JObject

object Single extends JsonDeserializer[SingleProcessorsChoice, JObject] {

  def deserialize(jObject: JObject): SingleProcessorsChoice = {
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
