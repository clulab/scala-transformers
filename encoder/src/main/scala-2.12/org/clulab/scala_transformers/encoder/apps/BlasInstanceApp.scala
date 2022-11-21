package org.clulab.scala_transformers.encoder.apps

import dev.ludovic.netlib.blas.{BLAS, JavaBLAS, NativeBLAS}

import scala.util.Try

object BlasInstanceApp extends App {
  Seq(
    (classOf[NativeBLAS], () => NativeBLAS.getInstance),
    (classOf[JavaBLAS],  () => JavaBLAS.getInstance),
    (classOf[BLAS], () => BLAS.getInstance)
  ).foreach { case (clazz, getInstance) =>
    val instanceTry = Try { getInstance() }

    if (instanceTry.isSuccess)
      println(s"Blas $clazz produced an instance of ${instanceTry.get.getClass.getName}.")
    else
      println(s"Blas $clazz produced no instance.")
  }
}
