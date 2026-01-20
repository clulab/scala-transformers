package org.clulab.scala_transformers.tokenizer.test

import java.io.File
import java.nio.file.Path
import scala.collection.JavaConverters._
import scala.util.{Try, Using}

class PythonProcess(pythonCwd: String, pythonCmd: String, pythonDir: String, pythonFile: String, arguments: Seq[String] = Seq.empty) {
  require(pythonFile.endsWith(PythonProcess.extension))
  val pythonCwdFile = Path.of(pythonCwd).toFile
  require(pythonCwdFile.exists)
  val pythonCmdFile = Path.of(pythonCwd).resolve(pythonCmd).toFile
  require(pythonCmdFile.exists)
  val pythonDirFile = Path.of(pythonCwd).resolve(pythonDir).toFile
  require(pythonDirFile.exists)
  val pythonFileFile = pythonDirFile.toPath.resolve(pythonFile).toFile
  require(pythonFileFile.exists)
  val errorFile = new File("error.txt")

  val pipeOpt = {
    val commands = (pythonCmd +: pythonFileFile.getPath +: arguments).asJava
    val processBuilder = new ProcessBuilder(commands)
        .directory(pythonCwdFile)
        .redirectError(errorFile)
    val processTry = Try(processBuilder.start)
    val processOpt = processTry.toOption
    val pipeOpt = processOpt.map { process =>
      new StdPipe(process.getInputStream, process.getOutputStream, process.getErrorStream)
    }

    if (processOpt.isEmpty) {
      val header = "Python"
      val content = "The Python process could not be started."

      println(s"$header: $content")
      //      new BusyWorker("Habitus", None).showWarning("Chat", header, content)
    }
    pipeOpt
  }

  def process(string: String): String = {
    pipeOpt.map { pipe =>
      pipe.write(string)
      pipe.read()
    }.getOrElse("")
  }
}

object PythonProcess {
  val extension = ".py"
}
