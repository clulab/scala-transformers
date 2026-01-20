package org.clulab.scala_transformers.tokenizer.test

object ThreadUtils {

  def addShutdownHook(block: => Unit): Unit = {
    Runtime.getRuntime.addShutdownHook(new Thread(new Runnable() {
      def run(): Unit = { block }
    }))
  }
}
