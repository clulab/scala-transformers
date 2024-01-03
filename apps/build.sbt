name := "scala-transformers-apps"
description := "Houses apps, particularly those having extra library dependencies"

resolvers ++= Seq(
  "Artifactory" at "https://artifactory.clulab.org/artifactory/sbt-release"
)

libraryDependencies ++= {
  Seq(
    // Models version 0.1.0 work when LinearLayer.USE_CONCAT == true.
    "org.clulab"     % "deberta-onnx-model" % "0.1.0",
    "org.clulab"     % "roberta-onnx-model" % "0.1.0",

    // Models version 0.2.0 work when LinearLayer.USE_CONCAT == false.
    // Models of different versions cannot be combined into a single project
    // because the resource names will conflict.  The choice is forced.
    // "org.clulab"     % "deberta-onnx-model" % "0.2.0",
    // "org.clulab"     % "electra-onnx-model" % "0.2.0",
    // "org.clulab"     % "roberta-onnx-model" % "0.2.0",

    "org.scalatest" %% "scalatest"          % "3.2.15" % "test"
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")
