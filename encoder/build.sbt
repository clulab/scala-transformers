name := "scala-transformers-encoder"
description := "Provides a Scala interface to huggingface encoders"

resolvers ++= Seq(
  "Artifactory" at "https://artifactory.clulab.org/artifactory/sbt-release"
)

libraryDependencies ++= {
  val breezeVersion = CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, minor)) if minor < 12 => "1.0"
    case _ => "2.1.0"
  }

  Seq(
    "org.scalanlp"              %% "breeze"             % breezeVersion,
    "com.microsoft.onnxruntime"  % "onnxruntime"        % "1.13.1",
    "org.clulab"                 % "roberta-onnx-model" % "0.0.1",
    "org.slf4j"                  % "slf4j-api"          % "1.7.10"
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
