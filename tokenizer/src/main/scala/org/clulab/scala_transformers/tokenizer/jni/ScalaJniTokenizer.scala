package org.clulab.scala_transformers.tokenizer.jni

import org.clulab.scala_transformers.tokenizer.LibraryLoader
import org.clulab.scala_transformers.tokenizer.Tokenization
import org.clulab.scala_transformers.tokenizer.Tokenizer

import java.io.File
import java.nio.file.{Files, Paths}
import scala.collection.mutable.{HashMap => MutableHashMap}
import scala.io.Source
import scala.ref.WeakReference

class ScalaJniTokenizer(name: String, addPrefixSpace: Boolean = false) extends Tokenizer(name) {
  val tokenizerId: Long = {
    // Try to load it from a file, which would override the resource.
    val file = new File(name)

    if (file.exists() && file.isFile) {
      val tokenizerId = JavaJniTokenizer.createFromFile(name)

      if (tokenizerId == 0)
        throw new RuntimeException(s"""The "$name" tokenizer could not be created by Tokenizer::from_file()!""")
      tokenizerId
    }
    // Try to load it from a resource if it is there.
    else {
      val resourceName = s"/org/clulab/scala_transformers/tokenizer/$name/tokenizer.json"
      val resourceOpt = Option(this.getClass.getResource(resourceName))

      if (resourceOpt.isDefined) {
        val resource = resourceOpt.get
        val bytes = Files.readAllBytes(Paths.get(resource.toURI))
        val tokenizerId = JavaJniTokenizer.createFromBytes(bytes)

        if (tokenizerId == 0)
          throw new RuntimeException(s"""The "$name" tokenizer could not be created by Tokenizer::from_pretrained()!""")
        tokenizerId
      }
      // As a last resort, try to fetch it over the network.
      else {
        val tokenizerId = JavaJniTokenizer.createFromPretrained(name)

        if (tokenizerId == 0)
          throw new RuntimeException(s"""The "$name" tokenizer could not be created by Tokenizer::from_pretrained()!""")
        tokenizerId
      }
    }
  }

  override def finalize(): Unit = {
    JavaJniTokenizer.destroy(tokenizerId)
  }

  override def tokenize(words: Array[String]): Tokenization = {
    val tokenizedWords =
        if (addPrefixSpace) words.map(" " + _)
        else words
    val javaJniTokenization = JavaJniTokenizer.tokenize(tokenizerId, tokenizedWords)

    Tokenization(
      javaJniTokenization.tokenIds,
      javaJniTokenization.wordIds,
      javaJniTokenization.tokens
    )
  }
}

object ScalaJniTokenizer {
  val rustLibrary: Unit = LibraryLoader.load("rust_tokenizer")
  val map = new MutableHashMap[(String, Boolean), WeakReference[ScalaJniTokenizer]]()

  def apply(name: String, addPrefixSpace: Boolean = false): ScalaJniTokenizer = synchronized {
    val key = (name, addPrefixSpace)
    // If the key is known and the weak reference is valid, then the result is
    // Some(scalaTokenizer) with a strong reference that will remain valid.
    val scalaTokenizerOpt = map.get(key).flatMap(_.get)

    if (scalaTokenizerOpt.isDefined)
      scalaTokenizerOpt.get
    else {
      val scalaTokenizer = new ScalaJniTokenizer(name, addPrefixSpace)
      map(key) = WeakReference(scalaTokenizer)
      scalaTokenizer
    }
  }
}
