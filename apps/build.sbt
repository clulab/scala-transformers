name := "scala-transformers-apps"
description := "Houses apps, particularly those having extra library dependencies"

resolvers ++= Seq(
  "Artifactory" at "https://artifactory.clulab.org/artifactory/sbt-release"
)

libraryDependencies ++= {
  Seq(
    "org.clulab"     % "roberta-onnx-model" % "0.1.0",
    "org.clulab"     % "deberta-onnx-model" % "0.1.0",
    "org.scalatest" %% "scalatest"          % "3.2.15" % "test"
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
