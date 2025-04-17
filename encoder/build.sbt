name := "maven-scala-transformers-encoder"
description := "Provides a Scala interface to huggingface encoders"

resolvers ++= Seq(
//  "Artifactory" at "https://artifactory.clulab.org/artifactory/sbt-release"
)

libraryDependencies ++= {
  val breezeVersion = CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, minor)) if minor < 12 => "1.0"
    case _ => "2.1.0"
  }
  val ejmlVersion = "0.41" // Use this older version for Java 8.

  Seq(
    // Choose one of these.
    /// "org.apache.commons"         % "commons-math3"      % "3.6.1",
    "org.ejml"                   % "ejml-core"          % ejmlVersion,
    "org.ejml"                   % "ejml-fdense"        % ejmlVersion,
    "org.ejml"                   % "ejml-simple"        % ejmlVersion,
    // "org.scalanlp"              %% "breeze"             % breezeVersion,

    "com.microsoft.onnxruntime"  % "onnxruntime"        % "1.13.1",
    "org.slf4j"                  % "slf4j-api"          % "1.7.10"
  )
}

fork := true

// assembly / mainClass := Some("com.keithalcock.tokenizer.scalapy.apps.ExampleApp")

enablePlugins(ShadingPlugin)
shadedDependencies ++= Set(
  "org.ejml" % "ejml-core"   % "<ignored>",
  "org.ejml" % "ejml-fdense" % "<ignored>",
  "org.ejml" % "ejml-simple" % "<ignored>"
)
shadingRules ++= Seq(
  ShadingRule.moveUnder("org.ejml", "org.clulab.shaded"),
  ShadingRule.moveUnder("pabeles.concurrency", "org.clulab.shaded")
)
validNamespaces ++= Set("org", "org.clulab")
