package org.clulab.scala_transformers.tokenizer.scala.tokenizers

import org.clulab.scala_transformers.tokenizer.Tokenization
import org.clulab.scala_transformers.tokenizer.scala.decoders.Decoder
import org.clulab.scala_transformers.tokenizer.scala.models.unigram.Unigram
import org.clulab.scala_transformers.tokenizer.scala.normalizers.NormalizerChoice
import org.clulab.scala_transformers.tokenizer.scala.postprocessors.PostProcessorChoice
import org.clulab.scala_transformers.tokenizer.scala.pretokenizers.PreTokenizerChoice
import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.json4s.jackson.JsonMethods
import org.json4s.{JArray, JObject, JValue}

import scala.io.{Codec, Source}
import scala.util.Using

class TruncationParams()

object TruncationParams extends JsonDeserializer[Option[TruncationParams], JValue] {

  def deserialize(jValue: JValue): Option[TruncationParams] = {
    val jObjectOpt = jValue.extractOpt[JObject]

    jObjectOpt.map(_ => new TruncationParams())
  }
}

class PaddingParams()

object PaddingParams extends JsonDeserializer[Option[PaddingParams], JValue] {

  def deserialize(jValue: JValue): Option[PaddingParams] = {
    val jObjectOpt = jValue.extractOpt[JObject]

    jObjectOpt.map(_ => new PaddingParams())
  }
}

class Token(
    val id: Int, // TODO: These IDs can be checked with tokenizer.token_to_id(&token.token.content).
    val content: String, // AddedToken
    val singleWord: Boolean, // AddedTokenOptions
    val lstrip: Boolean, // AddedTokenOptions
    val rstrip: Boolean, // AddedTokenOptions
    val normalized: Boolean, // AddedTokenOptions
    val special: Boolean // AddedToken
) {

}

object Token extends JsonDeserializer[Token, JObject] {

  def deserialize(jObject: JObject): Token = {
    val id = (jObject \ "id").extract[Int]
    val content = (jObject \ "content").extract[String]
    val singleWord = (jObject \ "single_word").extract[Boolean]
    val lstrip = (jObject \ "lstrip").extract[Boolean]
    val rstrip = (jObject \ "rstrip").extract[Boolean]
    val normalized = (jObject \ "normalized").extract[Boolean]
    val special = (jObject \ "special").extract[Boolean]

    new Token(id, content, singleWord, lstrip, rstrip, normalized, special)
  }
}

class JsonTokenizer(
  val version: String,
  val truncationOpt: Option[TruncationParams],
  val paddingOpt: Option[PaddingParams],
  val addedTokens: Seq[Token],
  val normalizer: NormalizerChoice,
  val preTokenizer: PreTokenizerChoice,
  val postProcessor: PostProcessorChoice,
  val decoder: Decoder,
  val model: Unigram
) {

}

object JsonTokenizer extends JsonDeserializer[JsonTokenizer, JObject] {
  val version = "1.0"

  def deserialize(jObject: JObject): JsonTokenizer = {
    val version = (jObject \ "version").extract[String]
    val truncationOpt = TruncationParams.deserialize(jObject \ "truncation")
    val paddingOpt = PaddingParams.deserialize(jObject \ "padding")
    val addedTokens = (jObject \ "added_tokens").extract[JArray].arr.map { jValue =>
      Token.deserialize(jValue.extract[JObject])
    }
    val normalizer = NormalizerChoice.deserialize((jObject \ "normalizer").extract[JObject])
    val preTokenizer = PreTokenizerChoice.deserialize((jObject \ "pre_tokenizer").extract[JObject])
    val postProcessor = PostProcessorChoice.deserialize((jObject \ "post_processor").extract[JObject])
    val decoder = Decoder.deserialize((jObject \ "decoder").extract[JObject])
    val model = Unigram.deserialize((jObject \ "model").extract[JObject])

    require(version == this.version)
    require(truncationOpt.isEmpty)
    require(paddingOpt.isEmpty)
    new JsonTokenizer(version, truncationOpt, paddingOpt, addedTokens,
        normalizer, preTokenizer, postProcessor, decoder, model)
  }
}

class MicrosoftDebertaV3BaseTokenizer(val name: String, val jsonTokenizer: JsonTokenizer) extends ScalaTokenizer(name) {

  // Get the name of the tokenizer.json file to read in.
  override def tokenize(words: Array[String]): Tokenization = {
    ???
  } // jsonTokenizer.tokenize(words)
}

object MicrosoftDebertaV3BaseTokenizer extends ScalaTokenizerConstructor with JsonDeserializer[MicrosoftDebertaV3BaseTokenizer, JObject] {
  val name = "microsoft/deberta-v3-base"

  protected def getTextFromResource(path: String): String = {
    Using.resource(Source.fromResource(path, getClass.getClassLoader)(Codec.UTF8)) { source =>
      source.mkString
    }
  }

  def deserialize(jObject: JObject): MicrosoftDebertaV3BaseTokenizer = {
    val jsonTokenizer = JsonTokenizer.deserialize(jObject)

    new MicrosoftDebertaV3BaseTokenizer(name, jsonTokenizer)
  }

  def construct(): MicrosoftDebertaV3BaseTokenizer = {
    val json = getTextFromResource(resourceName(name))
    val jObject = JsonMethods.parse(json).extract[JObject]

    deserialize(jObject)
  }
}
