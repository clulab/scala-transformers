package org.clulab.scala_transformers.tokenizer.jni

import org.clulab.scala_transformers.tokenizer.LibraryLoader
import org.clulab.scala_transformers.tokenizer.Tokenization
import org.clulab.scala_transformers.tokenizer.Tokenizer

import scala.collection.mutable.{HashMap => MutableHashMap}
import scala.ref.WeakReference

class ScalaJniTokenizer(name: String) extends Tokenizer(name) {
  val tokenizerId = JavaJniTokenizer.create(name)

  override def finalize: Unit = {
    JavaJniTokenizer.destroy(tokenizerId)
  }

  override def tokenize(words: Array[String]): Tokenization = {
    val javaJniTokenization = JavaJniTokenizer.tokenize(tokenizerId, words)

    Tokenization(
      javaJniTokenization.tokenIds,
      javaJniTokenization.wordIds,
      javaJniTokenization.tokens
    )
  }
}

object ScalaJniTokenizer {
  val rustLibrary = LibraryLoader.load("rust_tokenizer")
  val map = new MutableHashMap[String, WeakReference[ScalaJniTokenizer]]()

  def apply(name: String): ScalaJniTokenizer = synchronized {
    // If the key is known and the weak reference is valid, then the result is
    // Some(scalaTokenizer) with a strong reference that will remain valid.
    val scalaTokenizerOpt = map.get(name).flatMap(_.get)

    if (scalaTokenizerOpt.isDefined)
      scalaTokenizerOpt.get
    else {
      val scalaTokenizer = new ScalaJniTokenizer(name)
      map(name) = WeakReference(scalaTokenizer)
      scalaTokenizer
    }
  }
}
