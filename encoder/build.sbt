name := "scala-transformers-encoder"
description := "Provides a Scala interface to huggingface encoders"

resolvers ++= Seq(
//  Resolvers.localResolver, // Reserve for Two Six.
//  Resolvers.clulabResolver // processors-models, transitive dependency
)

libraryDependencies ++= {
  Seq(
    "com.microsoft.onnxruntime"  % "onnxruntime" % "1.13.1",
    "org.scalanlp" %% "breeze" % "2.1.0",
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")