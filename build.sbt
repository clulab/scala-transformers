val scala211 = "2.11.12" // up to 2.11.12
val scala212 = "2.12.18" // up to 2.12.18
val scala213 = "2.13.12" // up to 2.13.12
val scala30  = "3.0.2"   // up to 3.0.2
val scala31  = "3.1.3"   // up to 3.1.3
val scala32  = "3.2.2"   // up to 3.2.2
val scala33  = "3.3.1"   // up to 3.3.1
val scala3   = scala31

// Breeze 1.1+ is not available for scala211.
ThisBuild / crossScalaVersions := Seq(scala212, scala211, scala213, scala3)
ThisBuild / scalaVersion := scala212

name := "maven-scala-transformers"

lazy val root = (project in file("."))
  .aggregate(apps, common, tokenizer, encoder)
  .settings(
    publish / skip := true
  )

lazy val apps = project
  .dependsOn(encoder)
  .settings(
    publish / skip := true
  )

lazy val common = project

lazy val tokenizer = project
  .dependsOn(common % "compile -> compile; test -> test")

lazy val encoder = project
  .dependsOn(tokenizer % "compile -> compile; test -> test")
