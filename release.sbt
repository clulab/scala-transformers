import ReleaseTransformations._

// Latest version numbers were updated on 20216-04-24.
val scala211 = "2.11.12" // up to 2.11.12
val scala212 = "2.12.21" // up to 2.12.21
val scala213 = "2.13.18" // up to 2.13.18
val scala31  = "3.1.3"   // up to 3.1.3
// Only the LTS versions are listed next.
val scala33  = "3.3.7"   // up to 3.3.7
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

/*
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
//tagRelease,
  releaseStepCommandAndRemaining("+publishSigned"),   
  releaseStepCommandAndRemaining("sonaUpload"),     // log on to publish
  // releaseStepCommandAndRemaining("sonaRelease"), // automatically publish
//setNextVersion,
//commitNextVersion,
  // pushChanges
)
*/

releaseProcess := Seq[ReleaseStep](
  inquireVersions,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  setNextVersion,
  commitNextVersion,
//  pushChanges
)

Global / onChangedBuildSource := ReloadOnSourceChanges
Global / useGpg := false // GPG doesn't need to be installed, particularly for Windows.
