package org.clulab.scala_transformers.tokenizer.scala

import org.clulab.scala_transformers.tokenizer.scala.utils.JsonDeserializer
import org.clulab.scala_transformers.tokenizer.{Tokenization, Tokenizing}
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, JArray, JObject, JValue}

import scala.io.{Codec, Source}
import scala.util.Using

class Vocab(val text: String, val score: Double)

object Vocab extends JsonDeserializer[Vocab, JArray] {

  def deserialize(jArray: JArray): Vocab = {
    val arr = jArray.arr

    assert(arr.length == 2)

    val text = arr(0).extract[String]
    val score = arr(1).extract[Double]

    new Vocab(text, score)
  }
}

class Model(val unkId: Int, val vocab: Seq[Vocab]) {
  val typ = Model.typ
}

object Model extends JsonDeserializer[Model, JObject] {
  val typ = "Unigram"

  def deserialize(jObject: JObject): Model = {
    val typ = (jObject \ "type").extract[String]
    val unkId = (jObject \ "unk_id").extract[Int]
    val vocab = (jObject \ "vocab").extract[JArray].arr.map { jValue =>
      Vocab.deserialize(jValue.extract[JArray])
    }

    require(typ == this.typ)
    new Model(unkId, vocab)
  }
}

class Decoder(val replacement: String, val addPrefixSpace: Boolean) {
  val typ = Decoder.typ
}

object Decoder extends JsonDeserializer[Decoder, JObject] {
  val typ = "Metaspace"

  def deserialize(jObject: JObject): Decoder = {
    val typ = (jObject \ "type").extract[String]
    val replacement = (jObject \ "replacement").extract[String]
    val addPrefixSpace = (jObject \ "add_prefix_space").extract[Boolean]

    require(typ == this.typ)
    new Decoder(replacement, addPrefixSpace)
  }
}

class SpecialTokens(id: String, ids: Seq[Int], tokens: Seq[String])

object SpecialTokens extends JsonDeserializer[SpecialTokens, JObject] {

  def deserialize(jObject: JObject): SpecialTokens = {
    val id = (jObject \ "id").extract[String]
    val ids = (jObject \ "ids").extract[JArray].arr.map { jValue =>
      jValue.extract[Int]
    }
    val tokens = (jObject \ "tokens").extract[JArray].arr.map { jValue =>
      jValue.extract[String]
    }

    new SpecialTokens(id, ids, tokens)
  }
}

trait SingleItem
trait PairItem

class SpecialToken(val id: String, val typeId: Int) extends SingleItem with PairItem

object SpecialToken extends JsonDeserializer[SpecialToken, JObject] {

  def deserialize(jObject: JObject): SpecialToken = {
    val id = (jObject \ "id").extract[String]
    val typeId = (jObject \ "type_id").extract[Int]

    new SpecialToken(id, typeId)
  }
}

class Sequence(val id: String, val typeId: Int) extends SingleItem with PairItem

object Sequence extends JsonDeserializer[Sequence, JObject] {

  def deserialize(jObject: JObject): Sequence = {
    val id = (jObject \ "id").extract[String]
    val typeId = (jObject \ "type_id").extract[Int]

    new Sequence(id, typeId)
  }
}

class PostProcessor(val single: Seq[SingleItem], val pair: Seq[PairItem], val specialTokens: Map[String, SpecialTokens]) {
  val typ = PostProcessor.typ
}

object PostProcessor extends JsonDeserializer[PostProcessor, JObject] {
  val typ = "TemplateProcessing"

  def deserialize(jObject: JObject): PostProcessor = {
    val typ = (jObject \ "type").extract[String]
    val single = (jObject \ "single").extract[JArray].arr.map { jValue =>
      val jObject = jValue.extract[JObject]
      val specialTokenOpt = (jObject \ "SpecialToken").extractOpt[JObject].map { jObject =>
        SpecialToken.deserialize(jObject)
      }
      val sequenceOpt = (jObject \ "Sequence").extractOpt[JObject].map { jObject =>
        Sequence.deserialize(jObject)
      }

      (specialTokenOpt, sequenceOpt) match {
        case (Some(specialToken), None) => specialToken
        case (None, Some(sequence)) => sequence
        case _ => throw new RuntimeException("Could not deserialize the PostProcessor.")
      }
    }
    val pair = (jObject \ "pair").extract[JArray].arr.map { jValue =>
      val jObject = jValue.extract[JObject]
      val specialTokenOpt = (jObject \ "SpecialToken").extractOpt[JObject].map { jObject =>
        SpecialToken.deserialize(jObject)
      }
      val sequenceOpt = (jObject \ "Sequence").extractOpt[JObject].map { jObject =>
        Sequence.deserialize(jObject)
      }

      (specialTokenOpt, sequenceOpt) match {
        case (Some(specialToken), None) => specialToken
        case (None, Some(sequence)) => sequence
        case _ => throw new RuntimeException("Could not deserialize the PostProcessor.")
      }
    }
    val specialTokens = (jObject \ "special_tokens").extract[JObject].obj.map { jField =>
      val key = jField._1
      val jObject = jField._2.extract[JObject]
      val specialTokens = SpecialTokens.deserialize(jObject)

      key -> specialTokens
    }.toMap

    require(typ == this.typ)
    new PostProcessor(single, pair, specialTokens)
  }
}

trait PreTokenizerItem

class Metaspace(val replacement: String, val addPrefixSpace: Boolean) extends PreTokenizerItem {
  val typ = Metaspace.typ
}

object Metaspace extends JsonDeserializer[Metaspace, JObject] {
  val typ = "Metaspace"

  def deserialize(jObject: JObject): Metaspace = {
    val typ = (jObject \ "type").extract[String]
    val replacement = (jObject \ "replacement").extract[String]
    val addPrefixSpace = (jObject \ "add_prefix_space").extract[Boolean]

    require(typ == this.typ)
    new Metaspace(replacement, addPrefixSpace)
  }
}

class PreTokenizerGroup(val pretokenizers: Seq[PreTokenizerItem]) {
  val typ = PreTokenizerGroup.typ
}

object PreTokenizerGroup extends JsonDeserializer[PreTokenizerGroup, JObject] {
  val typ = "Sequence"

  def deserialize(jObject: JObject): PreTokenizerGroup = {
    val typ = (jObject \ "type").extract[String]
    val pretokenizers = (jObject \ "pretokenizers").extract[JArray].arr.map { jValue =>
      val jObject = jValue.extract[JObject]
      val typ = (jObject \ "type").extract[String]

      typ match {
        case Metaspace.typ => Metaspace.deserialize(jObject)
        case _ => throw new RuntimeException("Could not deserialize PreTokenizerGroup.")
      }
    }
    require(typ == this.typ)
    new PreTokenizerGroup(pretokenizers)
  }
}

class Pattern(val regex: String)

object Pattern extends JsonDeserializer[Pattern, JObject] {

  def deserialize(jObject: JObject): Pattern = {
    val regex = (jObject \ "Regex").extract[String]

    new Pattern(regex)
  }
}

trait NormalizerItem

class Strip(val stripLeft: Boolean, val stripRight: Boolean) extends NormalizerItem {
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

class Precompiled(val precompiledCharsmap: String) extends NormalizerItem {
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

class Replace(val pattern: Pattern, val content: String) extends NormalizerItem {
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

class Normalizer(val normalizers: Seq[NormalizerItem]) {
  val typ = Normalizer.typ
}

object Normalizer extends JsonDeserializer[Normalizer, JObject] {
  val typ = "Sequence"

  def deserialize(jObject: JObject): Normalizer = {
    val typ = (jObject \ "type").extract[String]
    val normalizers = (jObject \ "normalizers").extract[JArray].arr.map { jValue =>
      val jObject = jValue.extract[JObject]
      val typ = (jObject \ "type").extract[String]

      typ match {
        case Strip.typ => Strip.deserialize(jObject)
        case Replace.typ => Replace.deserialize(jObject)
        case Precompiled.typ => Precompiled.deserialize(jObject)
        case _ => throw new RuntimeException("Could not deserialize Normalizer.")
      }
    }

    require(typ == this.typ)
    new Normalizer(normalizers)
  }
}

class Token(
  val id: Int,
  val content: String,
  val singleWord: Boolean,
  val lstrip: Boolean,
  val rstrip: Boolean,
  val normalized: Boolean,
  val special: Boolean
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
  val truncationOpt: Option[Boolean],
  val paddingOpt: Option[Boolean],
  val addedTokens: Seq[Token],
  val normalizer: Normalizer,
  val preTokenizer: PreTokenizerGroup,
  val postProcessor: PostProcessor,
  val decoder: Decoder,
  val model: Model
) {

}

object JsonTokenizer extends JsonDeserializer[JsonTokenizer, JObject] {

  def deserialize(jObject: JObject): JsonTokenizer = {
    val version = (jObject \ "version").extract[String]
    val truncationOpt = (jObject \ "truncation").extractOpt[Boolean]
    val paddingOpt = (jObject \ "padding").extractOpt[Boolean]
    val addedTokens = (jObject \ "added_tokens").extract[JArray].arr.map { jValue =>
      Token.deserialize(jValue.extract[JObject])
    }
    val normalizer = Normalizer.deserialize((jObject \ "normalizer").extract[JObject])
    val preTokenizer = PreTokenizerGroup.deserialize((jObject \ "pre_tokenizer").extract[JObject])
    val postProcessor = PostProcessor.deserialize((jObject \ "post_processor").extract[JObject])
    val decoder = Decoder.deserialize((jObject \ "decoder").extract[JObject])
    val model = Model.deserialize((jObject \ "model").extract[JObject])
    val jsonTokenizer = new JsonTokenizer(version, truncationOpt, paddingOpt, addedTokens,
        normalizer, preTokenizer, postProcessor, decoder, model)

    jsonTokenizer
  }
}

class MicrosoftDebertaV3BaseTokenizer(val name: String, val jsonTokenizer: JsonTokenizer) extends ScalaTokenizer(name) {

  // Get the name of the tokenizer.json file to read in.
  override def tokenize(words: Array[String]): Tokenization = null // jsonTokenizer.tokenize(words)
}

object MicrosoftDebertaV3BaseTokenizer extends ScalaTokenizerConstructor {
  implicit val formats: DefaultFormats.type = DefaultFormats
  val name = "microsoft/deberta-v3-base"

  protected def getTextFromResource(path: String): String = {
    Using.resource(Source.fromResource(path, getClass.getClassLoader)(Codec.UTF8)) { source =>
      source.mkString
    }
  }

  def construct(): MicrosoftDebertaV3BaseTokenizer = {
    val json = getTextFromResource(resourceName(name))
    val jObject = JsonMethods.parse(json).extract[JObject]
    val jsonTokenizer = JsonTokenizer.deserialize(jObject)

    new MicrosoftDebertaV3BaseTokenizer(name, jsonTokenizer)
  }
}
