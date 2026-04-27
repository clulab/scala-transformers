// Latest version numbers were updated on 2026-04-24.
val scala211 = "2.11.12" // up to 2.11.12
val scala212 = "2.12.21" // up to 2.12.21
val scala213 = "2.13.18" // up to 2.13.18
val scala31  = "3.1.3"   // up to 3.1.3
// Only the LTS versions are listed next.
val scala33  = "3.3.7"   // up to 3.3.7
val scala3   = scala31

ThisBuild / crossScalaVersions := Seq(scala212, scala211, scala213, scala3)
ThisBuild / scalaVersion := scala212
ThisBuild / versionScheme := Some("early-semver")

name := "scala-transformers"

lazy val root = (project in file("."))
  .aggregate(apps, common, tokenizer, encoder, resources)
  .settings(
    crossScalaVersions := Nil,
    publish / skip := true
  )

lazy val apps = project
  .dependsOn(encoder)
  .settings(
    publish / skip := true
  )

lazy val common = project

lazy val resources = project
  .settings(
    crossPaths := false, // This is a resource only and is independent of Scala version.
    autoScalaLibrary := false,
    Compile / packageBin / publishArtifact := true,  // Do include the resources.
    Compile / packageDoc / publishArtifact := false, // There is no documentation.
    Compile / packageSrc / publishArtifact := false, // There is no source code.
    Test / packageBin / publishArtifact := false,
    // Let’s remove any repositories for optional dependencies in our artifact.
    pomIncludeRepository := { _ => false }
  )

lazy val tokenizer = project
  .dependsOn(common % "compile -> compile; test -> test", resources)
  .settings(
//    publish / skip := true // This is too large to publish reliably.  Use extra release commands.
  )  
 
lazy val encoder = project
  .dependsOn(tokenizer % "compile -> compile; test -> test")
