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
}

trait JsonIndexedDeserializer[T, J <: JValue] {
  implicit val formats: DefaultFormats.type = DefaultFormats

  def deserialize(index: Int, jValue: J): T = deserializeOpt(index, jValue).get

  def deserializeOpt(index: Int, jValue: J): Option[T]
}
