import ReleaseTransformations._

ThisBuild / credentials ++= {
  val file = Path.userHome / ".sbt" / ".clulab-credentials"

  if (file.exists) Seq(Credentials(file))
  else Seq.empty
}

releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  releaseStepCommandAndRemaining("+publishSigned"),
  setNextVersion,
  commitNextVersion,
  releaseStepCommandAndRemaining("sonatypeReleaseAll"),
  pushChanges
)
