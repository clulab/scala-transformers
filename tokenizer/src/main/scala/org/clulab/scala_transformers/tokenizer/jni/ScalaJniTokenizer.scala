package org.clulab.scala_transformers.tokenizer.jni

import org.clulab.scala_transformers.common.LibraryLoader
import org.clulab.scala_transformers.tokenizer.Tokenization
import org.clulab.scala_transformers.tokenizer.Tokenizer

import java.io.{ByteArrayOutputStream, File}
import scala.collection.mutable.{HashMap => MutableHashMap}
import scala.ref.WeakReference

class ScalaJniTokenizer(name: String, addPrefixSpace: Boolean = false) extends Tokenizer(name) {
  val tokenizerId: Long = {
    from_file(name)
      .getOrElse(from_resource(name)
        .getOrElse(from_network(name)
          .getOrElse(0)
        )
      )
  }

  protected def from_file(name: String): Option[Long] = {
    val file = new File(name)

    if (file.exists() && file.isFile) {
      val tokenizerId = JavaJniTokenizer.createFromFile(name)

      if (tokenizerId == 0)
        throw new RuntimeException(s"""The "$name" tokenizer could not be created by Tokenizer::from_file()!""")
      Some(tokenizerId)
    }
    else None
  }

  protected def from_resource(name: String): Option[Long] = {
    val resourceName = s"/org/clulab/scala_transformers/tokenizer/$name/tokenizer.json"
    val resourceURLOpt = Option(getClass.getResource(resourceName))

    resourceURLOpt.map { resourceURL =>
      val bufferSize = 10280
      val byteBuffer = new Array[Byte](bufferSize)
      val outputBuffer = new ByteArrayOutputStream()
      val inputStream = resourceURL.openStream()

      @annotation.tailrec
      def readBytes(): Unit = {
        val count = inputStream.read(byteBuffer)

        if (count >= 0) {
          outputBuffer.write(byteBuffer, 0, count)
          readBytes()
        }
      }

      try {
        readBytes()
      }
      finally {
        inputStream.close()
      }

      val bytes = outputBuffer.toByteArray
      val tokenizerId = JavaJniTokenizer.createFromBytes(bytes)

      if (tokenizerId == 0)
        throw new RuntimeException(s"""The "$name" tokenizer could not be created by Tokenizer::from_pretrained()!""")
      tokenizerId
    }
  }

  protected def from_network(name: String): Option[Long] = {
    val tokenizerId = JavaJniTokenizer.createFromPretrained(name)

    if (tokenizerId == 0)
      throw new RuntimeException(s"""The "$name" tokenizer could not be created by Tokenizer::from_pretrained()!""")
    Some(tokenizerId)
  }

  override def finalize(): Unit = {
    // With this design, it is possible to get a tokenizerId of 0.  An exception will
    // be thrown, but the object is still created and will be finalized.
    if (tokenizerId != 0)
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
