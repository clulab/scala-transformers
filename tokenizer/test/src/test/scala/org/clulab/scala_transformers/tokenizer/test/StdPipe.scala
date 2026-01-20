package org.clulab.scala_transformers.tokenizer.test

import java.io.{BufferedReader, InputStream, InputStreamReader, OutputStream, PrintStream}
import java.nio.charset.StandardCharsets
import scala.annotation.tailrec

class StdPipe(
    inputStream: InputStream,
    outputStream: OutputStream,
    errorStream: InputStream
) extends Pipe {
  val inputBufferedReader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))
  val inputBuffer = new Array[Char](StdPipe.inputBufferSize)
  val outputPrintStream = new PrintStream(outputStream, true, StandardCharsets.UTF_8.toString)
  //  val errorBufferedReader = new BufferedReader(new InputStreamReader(errorStream, StandardCharsets.UTF_8))
  //  val errorBuffer = new Array[Char](StdPipe.errorBuferSize)

  def write(text: String): Unit = {
    // For us, a line ends with NL.
//    val nlCount = text.count(_ == StdPipe.NL) + 1 // An extra will be added later.

    // The count should contain only digits, so println is fine.
//    outputPrintStream.println(nlCount)
    // This had better be in binary mode.
    // An extra NL is added at the end and should not be read back in.
    outputPrintStream.print(text)
    outputPrintStream.print(StdPipe.NL)
  }

  //  def readErr(): Unit =
  //
  //    @tailrec
  //    def loop(): Unit = {
  //      if (errorBufferedReader.ready()) {
  //        val readCount = errorBufferedReader.read(errorBuffer)
  //
  //        if (0 < readCount) {
  //          errorBuffer.take(readCount).foreach(print)
  //          loop()
  //        }
  //      }
  //    }
  //
  //    loop()
  //  }

  // TODO: Read more than one character at a time.
  def readIn(): String = {
    // Since there may have been single CRs in the text that need to be preserved
    // and readLine throws them away, we'll have to read one character at a time.
    // TODO: That's not really true.  Multiple can be read at a time.
    // It is OK to read too much.  We just have to make sure to read enough and to
    // keep waiting only when there is more to be expected.
    // val line = inputBufferedReader.readLine

    val line = {
      val textBuffer = new StringBuffer()

      while ({
        //        readErr()
        val readCount = inputBufferedReader.read(inputBuffer, 0, 1)

        if (0 < readCount) {
          textBuffer.append(inputBuffer.head)

          inputBuffer.head != StdPipe.NL
        }
        else if (0 == readCount) {
          Thread.sleep(1000)
          true
        }
        else {
          false
        }
      }) ()
      textBuffer.toString
    }
    val nlCount = line.trim.toInt
    val textBuffer = new StringBuffer()

    @tailrec
    def loop(counted: Int): Unit = {
      val readCount = inputBufferedReader.read(inputBuffer)

      if (0 <= readCount) {
        val newCounted = counted + inputBuffer.take(readCount).count(_ == StdPipe.NL)

        if (newCounted < nlCount) {
          inputBuffer.take(readCount).foreach(textBuffer.append)
          loop(newCounted)
        }
        else {
          inputBuffer.take(readCount - 1).foreach(textBuffer.append)

          while (inputBufferedReader.ready) {
            val readCount = inputBufferedReader.read(inputBuffer)

            inputBuffer.take(readCount).foreach(textBuffer.append)
          }
        }
      }
    }

    loop(0)
    textBuffer.toString
  }

  def read(): String = {
    //    readErr() // In case something happened in the meantime, check it out.
    readIn()
  }
}

object StdPipe {
  val NL = '\n'
  val CR = '\r'
  val errorBuferSize = 512
  val inputBufferSize = 1024
}
