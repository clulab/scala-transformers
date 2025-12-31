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
      case _ => None
    }
  }

  def mkDeserializationError(anyRef: AnyRef, name: String, value: String): RuntimeException = {
    val className = anyRef.getClass.getSimpleName

    throw new RuntimeException(s"""The $className could not deal with a "$name" of "$value".""")
  }
}
