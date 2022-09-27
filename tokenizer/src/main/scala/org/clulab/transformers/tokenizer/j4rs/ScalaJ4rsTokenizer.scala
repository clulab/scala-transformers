package org.clulab.transformers.tokenizer.j4rs

import org.clulab.transformers.tokenizer.LibraryLoader
import org.clulab.transformers.tokenizer.Tokenization
import org.clulab.transformers.tokenizer.Tokenizer

import scala.collection.mutable.{HashMap => MutableHashMap}
import scala.ref.WeakReference

class ScalaJ4rsTokenizer(name: String) extends Tokenizer(name) {
  val tokenizerId = JavaJ4rsTokenizer.create(name)

  override def finalize: Unit = {
    JavaJ4rsTokenizer.destroy(tokenizerId)
  }

  override def tokenize(words: Array[String]): Tokenization = {
    val javaJ4rsTokenization = JavaJ4rsTokenizer.tokenize(tokenizerId, words)

    Tokenization(
      javaJ4rsTokenization.tokenIds,
      javaJ4rsTokenization.wordIds,
      javaJ4rsTokenization.tokens
    )
  }
}

object ScalaJ4rsTokenizer {
  val rustLibrary = LibraryLoader.load("rust_tokenizer")
  val map = new MutableHashMap[String, WeakReference[ScalaJ4rsTokenizer]]()

  def apply(name: String): ScalaJ4rsTokenizer = synchronized {
    // If the key is known and the weak reference is valid, then the result is
    // Some(scalaTokenizer) with a strong reference that will remain valid.
    val scalaTokenizerOpt = map.get(name).flatMap(_.get)

    if (scalaTokenizerOpt.isDefined)
      scalaTokenizerOpt.get
    else {
      val scalaTokenizer = new ScalaJ4rsTokenizer(name)
      map(name) = WeakReference(scalaTokenizer)
      scalaTokenizer
    }
  }
}
