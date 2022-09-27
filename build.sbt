val scala211 = "2.11.12" // up to 2.11.12
val scala212 = "2.12.16" // up to 2.12.16
val scala213 = "2.13.8"  // up to 2.13.8
val scala30  = "3.0.2"   // up to 3.0.2
val scala31  = "3.1.3"   // up to 3.1.3

name := "tokenizer"

ThisBuild / scalaVersion := scala212

lazy val root = (project in file("."))
  .dependsOn(common % "compile -> compile; test -> test")
  .dependsOn(scalapy)

lazy val common = project

lazy val tokenizer = project
  .dependsOn(common % "compile -> compile; test -> test")
