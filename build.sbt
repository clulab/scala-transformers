val scala211 = "2.11.12" // up to 2.11.12
val scala212 = "2.12.17" // up to 2.12.17
val scala213 = "2.13.9"  // up to 2.13.9
val scala30  = "3.0.2"   // up to 3.0.2
val scala31  = "3.1.3"   // up to 3.1.3
val scala32  = "3.2.0"   // up to 3.2.0

ThisBuild / crossScalaVersions := Seq(scala212, scala211, scala213)
ThisBuild / scalaVersion := scala212

name := "scala-transformers"

lazy val root = (project in file("."))
  .aggregate(common, tokenizer, encoder)
  .dependsOn(common, tokenizer, encoder % "compile -> compile; test -> test")
  .settings(
    publish / skip := true
  )

lazy val common = project

lazy val tokenizer = project
  .dependsOn(common % "compile -> compile; test -> test")

lazy val encoder = project
  .dependsOn(tokenizer % "compile -> compile; test -> test")

