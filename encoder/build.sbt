name := "scala-transformers-encoder"
description := "Provides a Scala interface to huggingface encoders"

resolvers ++= Seq(
//  Resolvers.localResolver, // Reserve for Two Six.
//  Resolvers.clulabResolver // processors-models, transitive dependency
)

libraryDependencies ++= {
  Seq(
    "com.microsoft.onnxruntime"  % "onnxruntime" % "1.8.1",
    "org.scala-lang.modules" %% "scala-parser-combinators" % "1.0.6",
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
