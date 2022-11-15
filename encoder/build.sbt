name := "scala-transformers-encoder"
description := "Provides a Scala interface to huggingface encoders"

resolvers ++= Seq(
//  Resolvers.localResolver, // Reserve for Two Six.
//  Resolvers.clulabResolver // processors-models, transitive dependency
)

libraryDependencies ++= {
  Seq(
    "org.scalanlp"              %% "breeze"      % "1.0", //  "2.1.0",
    "com.microsoft.onnxruntime"  % "onnxruntime" % "1.13.1",
    "org.slf4j"                  % "slf4j-api"   % "1.7.10"
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
