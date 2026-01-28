package org.clulab.scala_transformers.tokenizer.test

import org.clulab.scala_transformers.tokenizer.jni.ScalaJniTokenizer
import org.clulab.scala_transformers.tokenizer.{Tokenization, Tokenizing}

class ExternalTokenizer(pythonProcess: PythonProcess) extends Tokenizing {

  override def tokenize(words: Array[String]): Tokenization = {
    val output = pythonProcess.process(words.mkString(" "))
    val lines = output.split("\n").map(_.replace("\r", ""))
    val tokenIds = lines(0).split(' ').map(_.toInt)
    val wordIds = lines(1).split(' ').map(_.toInt)
    val tokens = lines(2).split(' ')
    val tokenization = Tokenization(tokenIds, wordIds, tokens)

    tokenization
  }
}

class MultiTokenizer(name: String, addPrefixSpace: Boolean = false) extends Tokenizing {
//  println(new File(".").getAbsolutePath)
  val pythonCwd = "./"
//  val pythonCmd = ".venv/Scripts/python.exe"
  val pythonCmd = ".venv/bin/python3"
  val pythonDir = "tokenizer/test/src/test/python/"
  val pythonFile = "tokeneyes.py"

  val pythonTokenizer = {
    val pythonProcess = new PythonProcess(
      pythonCwd, pythonCmd,
      pythonDir, pythonFile,
      Seq(name, addPrefixSpace.toString, false.toString)
    )

    new ExternalTokenizer(pythonProcess)
  }
  val rustThruPythonTokenizer = {
    val pythonProcess = new PythonProcess(
      pythonCwd, pythonCmd,
      pythonDir, pythonFile,
      Seq(name, addPrefixSpace.toString, true.toString)
    )

    new ExternalTokenizer(pythonProcess)
  }
  // val scalaTokenizer = null
  val rustThruScalaTokenizer  = ScalaJniTokenizer(name, addPrefixSpace)
  val tokenizers = Seq(
    /*pythonTokenizer,*/ rustThruPythonTokenizer, rustThruScalaTokenizer
  )

  override def tokenize(words: Array[String]): Tokenization = {
    val tokenizations = tokenizers.map(_.tokenize(words))

    tokenizations.tail.foreach { tokenization =>
      tokenizations.head.tokenIds.zip(tokenization.tokenIds).foreach { case (left, right) =>
        left == right
      }
      tokenizations.head.wordIds.zip(tokenization.wordIds).foreach { case (left, right) =>
        left == right
      }
      tokenizations.head.tokens.zip(tokenization.tokens).foreach { case (left, right) =>
        left == right
      }
    }

    tokenizations.head
  }
}
