name := "scala-transformers-tokenizer-scala"
description := "Provides a tokenizer implemented in Scala without any Rust"

resolvers ++= Seq(
//  Resolvers.localResolver, // Reserve for Two Six.
//  Resolvers.clulabResolver // processors-models, transitive dependency
)

libraryDependencies ++= {
  val json4sVersion = "3.5.5"

  Seq(
    // JSON
    "org.json4s"             %% "json4s-core"             % json4sVersion,
    "org.json4s"             %% "json4s-jackson"          % json4sVersion
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
