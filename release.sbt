import ReleaseTransformations._

val scala211 = "2.11.12" // up to 2.11.12
val scala212 = "2.12.18" // up to 2.12.18
val scala213 = "2.13.12" // up to 2.13.12
val scala30  = "3.0.2"   // up to 3.0.2
val scala31  = "3.1.3"   // up to 3.1.3
val scala32  = "3.2.2"   // up to 3.2.2
val scala33  = "3.3.1"   // up to 3.3.1
val scala3   = scala31

val releaseStepCommands: Seq[String] = {
  val versions = Seq(scala212, scala211, scala213, scala3)
  val projects = Seq("common", "tokenizer", "encoder")
  val strings = versions.flatMap { version =>
    projects.map { project =>
      s"""clean; ++ $version; $project / publishSigned; $project / sonaBundle; sonaUpload"""
    }
  }
  
  println("\nYou may also need these commands to release:\n")
  strings.foreach(println)
  println()
  strings
}

releaseProcess := Seq[ReleaseStep](
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  releaseStepCommandAndRemaining("+publishSigned"),   
  releaseStepCommandAndRemaining("sonaUpload"),     // log on to publish
  // releaseStepCommandAndRemaining("sonaRelease"), // automatically publish
)

/*
releaseProcess := Seq[ReleaseStep](
  inquireVersions,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  setNextVersion,
  commitNextVersion,
//  pushChanges
)
*/
