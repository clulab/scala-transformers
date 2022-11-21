name := "scala-transformers-encoder"
description := "Provides a Scala interface to huggingface encoders"

resolvers ++= Seq(
//  Resolvers.localResolver, // Reserve for Two Six.
//  Resolvers.clulabResolver // processors-models, transitive dependency
)

libraryDependencies ++= {
  val breezeVersion = CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, minor)) if minor < 12 => "1.0"
    case _ => "2.1.0"
  }

  Seq(
    "org.scalanlp"              %% "breeze"      % breezeVersion,
    "com.microsoft.onnxruntime"  % "onnxruntime" % "1.13.1",
    "org.slf4j"                  % "slf4j-api"   % "1.7.10"
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
