package org.clulab.scala_transformers.tokenizer.scala

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.clulab.scala_transformers.tokenizer.{Tokenization, Tokenizing}
import org.json4s.{JArray, JObject}

class Vocab(text: String, score: Float)

object Vocab extends JsonDeserializer[Vocab, JArray] {

  def deserializeOpt(jObject: JArray): Option[Vocab] = {
    ???
  }
}

class Model(unkId: Int, vocab: Seq[Vocab]) {
  val typ = Model.typ
}

object Model extends JsonDeserializer[Model, JObject] {
  val typ = "Unigram"

  def deserializeOpt(jObject: JObject): Option[Model] = {
    ???
  }
}

class Decoder(replacement: String, addPrefixSpace: Boolean) {
  val typ = Decoder.typ
}

object Decoder extends JsonDeserializer[Decoder, JObject] {
  val typ = "Metaspace"

  def deserializeOpt(jObject: JObject): Option[Decoder] = {
    ???
  }
}

class SpecialTokens(id: String, ids: Seq[Int], tokens: Seq[String])

object SpecialTokens extends JsonDeserializer[SpecialTokens, JObject] {

  def deserializeOpt(jObject: JObject): Option[SpecialTokens] = {
    ???
  }
}

trait SingleItem

class SpecialToken(id: String, typeId: Int) extends SingleItem

object SpecialToken extends JsonDeserializer[SpecialToken, JObject] {

  def deserializeOpt(jObject: JObject): Option[SpecialToken] = {
    ???
  }
}

class Sequence(id: String, typeId: Int) extends SingleItem

object Sequence extends JsonDeserializer[Sequence, JObject] {

  def deserializeOpt(jObject: JObject): Option[Sequence] = {
    ???
  }
}

trait PairItem

class PostProcessor(single: Seq[SingleItem], pair: Seq[PairItem], specialTokens: Map[String, SpecialTokens]) {
  val typ = PostProcessor.typ
}

object PostProcessor extends JsonDeserializer[PostProcessor, JObject] {
  val typ = "TemplateProcessing"

  def deserializeOpt(jObject: JObject): Option[PostProcessor] = {
    ???
  }
}

trait PreTokenizerItem

class MetaspacePreTokenizer(replacement: String, addPrefixSpace: Boolean) extends PreTokenizerItem {
  val typ = MetaspacePreTokenizer.typ
}

object MetaspacePreTokenizer extends JsonDeserializer[MetaspacePreTokenizer, JObject] {
  val typ = "Metaspace"

  def deserializeOpt(jObject: JObject): Option[MetaspacePreTokenizer] = {
    ???
  }
}

class PreTokenizerGroup(pretokenizers: Seq[PreTokenizerItem]) {
  val typ = PreTokenizerGroup.typ
}

object PreTokenizerGroup extends JsonDeserializer[PreTokenizerGroup, JObject] {
  val typ = "Sequence"

  def deserializeOpt(jObject: JObject): Option[Pattern] = {
    ???
  }
}


class Pattern(regex: String)

object Pattern extends JsonDeserializer[Pattern, JObject] {

  def deserializeOpt(jObject: JObject): Option[Pattern] = {
    ???
  }
}

trait NormalizerItem

class StripNormalizer(stripLeft: Boolean, stripRight: Boolean) extends NormalizerItem {
  val typ = StripNormalizer.typ
}

object StripNormalizer extends JsonDeserializer[StripNormalizer, JObject] {
  val typ = "Strip"

  def deserializeOpt(jObject: JObject): Option[StripNormalizer] = {
    ???
  }
}

class PrecompiledNormalizer(precompiledCharsmap: String) extends NormalizerItem {
  val typ = PrecompiledNormalizer.typ
}

object PrecompiledNormalizer extends JsonDeserializer[PrecompiledNormalizer, JObject] {
  val typ = "Precompiled"

  def deserializeOpt(jObject: JObject): Option[PrecompiledNormalizer] = {
    ???
  }
}

class ReplaceNormalizer(pattern: Pattern, content: String) extends NormalizerItem {
  val typ = ReplaceNormalizer.typ
}

object ReplaceNormalizer extends JsonDeserializer[ReplaceNormalizer, JObject] {
  val typ = "Replace"

  def deserializeOpt(jObject: JObject): Option[ReplaceNormalizer] = {
    ???
  }
}

class NormalizerGroup

object NormalizerGroup extends JsonDeserializer[NormalizerGroup, JObject] {

  def deserializeOpt(jObject: JObject): Option[NormalizerGroup] = {
    ???
  }
}

class Token(
  id: Int,
  content: String,
  singleWord: Boolean,
  lstrip: Boolean,
  rstrip: Boolean,
  normalized: Boolean,
  special: Boolean
) {

}

object Token extends JsonDeserializer[Token, JObject] {

  def deserializeOpt(jObject: JObject): Option[Token] = {
    ???
  }
}


class JsonTokenizer(
  version: String,
  truncation: Option[Boolean],
  padding: Option[Boolean],
  addedTokens: Seq[Token],
  normalizer: NormalizerGroup,
  preTokenizer: PreTokenizer,
  postProecessor: PostProcessor,
  decoder: Decoder,
  model: Model
) {

}

object JsonTokenizer extends JsonDeserializer[JsonTokenizer, JObject] {

  def deserializeOpt(jObject: JObject): Option[JsonTokenizer] = {
    ???
  }
}

class MicrosoftDebertaV3BaseTokenizer(name: String) extends ScalaTokenizer(name) {

  // Read in the JSON stuff, get the particular kind of Tokenizer like MicrosoftDebertaV3Base()

  val tokenizer: Tokenizing = null // ScalaJniTokenizer(name, addPrefixSpace)

  // Get the name of the tokenizer.json file to read in.
  override def tokenize(words: Array[String]): Tokenization = tokenizer.tokenize(words)
}

object MicrosoftDebertaV3BaseTokenizer extends ScalaTokenizerConstructor {
  val name = "microsoft/deberta-v3-base"

  def construct(): MicrosoftDebertaV3BaseTokenizer = new MicrosoftDebertaV3BaseTokenizer(name)
}
