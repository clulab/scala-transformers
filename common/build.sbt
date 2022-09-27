name := "scala-transformers-common"
description := "Supports subprojects with common code"

resolvers ++= Seq(
//  Resolvers.localResolver, // Reserve for Two Six.
//  Resolvers.clulabResolver // processors-models, transitive dependency
)

libraryDependencies ++= {
  Seq(
	    "org.scalatest" %% "scalatest"           % "3.0.9" % "test"
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
