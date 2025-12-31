package org.clulab.scala_transformers.tokenizer.scala

import org.clulab.scala_transformers.tokenizer.Tokenizing

import scala.collection.mutable.{HashMap => MutableHashMap}
import scala.ref.WeakReference

abstract class ScalaTokenizer(name: String) extends Tokenizing

abstract class ScalaTokenizerConstructor {
  def construct(): ScalaTokenizer

  def resourceName(name: String) = s"org/clulab/scala_transformers/tokenizer/scala/$name/tokenizer.json"
}

object UnknownTokenizer extends ScalaTokenizerConstructor {
  def construct(): ScalaTokenizer = ???
}

object ScalaTokenizer {
  val constructors: Map[String, ScalaTokenizerConstructor] = Map(
    "etc." -> UnknownTokenizer,
    MicrosoftDebertaV3BaseTokenizer.name -> MicrosoftDebertaV3BaseTokenizer
  )
  val map = new MutableHashMap[String, WeakReference[ScalaTokenizer]]()

  def apply(name: String, addPrefixSpace: Boolean = true): ScalaTokenizer = synchronized {
    val expectedAddPrefixSpace: Boolean = name.contains("roberta")
    if (addPrefixSpace != expectedAddPrefixSpace)
      throw new IllegalArgumentException("The value for addPrefixSpace is not acceptable.")

    val key = name
    if (!constructors.contains(key))
      throw new IllegalArgumentException("The value for name is not acceptable.")

    // If the key is known and the weak reference is valid, then the result is
    // Some(scalaTokenizer) with a strong reference that will remain valid.
    val scalaTokenizerOpt = map.get(key).flatMap(_.get)

    scalaTokenizerOpt.getOrElse {
      val scalaTokenizer = constructors(key).construct()

      map(key) = WeakReference(scalaTokenizer)
      scalaTokenizer
    }
  }
}
