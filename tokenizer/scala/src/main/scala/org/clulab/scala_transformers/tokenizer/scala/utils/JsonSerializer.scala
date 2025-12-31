package org.clulab.scala_transformers.tokenizer.scala.utils

import org.json4s.{DefaultFormats, JValue}

trait JsonSerializer[T <: JValue] {
  implicit val formats: DefaultFormats.type = DefaultFormats

  def serialize: T
}

trait JsonDeserializer[T, J <: JValue] {
  implicit val formats: DefaultFormats.type = DefaultFormats

  def deserialize(jValue: J): T

  def deserializeOpt(jValue: J): Option[T] = {
    try {
      Some(deserialize(jValue))
    }
    catch {
      case _: Throwable => None
    }
  }

  def mkTypeError(anyRef: AnyRef, value: String): RuntimeException = {
    val className = anyRef.getClass.getSimpleName

    new RuntimeException(s"""The $className could not deal with a "type" of "$value".""")
  }

  def mkKeyError(anyRef: AnyRef, keys: String*): RuntimeException = {
    val className = anyRef.getClass.getSimpleName
    val keyStrings = keys.mkString("\"", "\", \"", "\"")

    new RuntimeException(s"""The $className could not find an entry for any of $keyStrings.""")
  }
}
