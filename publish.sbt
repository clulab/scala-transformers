import org.clulab.sbt.BuildUtils
import org.clulab.sbt.DependencyFilter
import org.clulab.sbt.DependencyId

import scala.xml.Node
import scala.xml.transform.RuleTransformer

val publication = "scala-transformers"

ThisBuild / developers := List(
  Developer(
    id    = "mihai.surdeanu",
    name  = "Mihai Surdeanu",
    email = "mihai@surdeanu.info",
    url   = url("https://www.cs.arizona.edu/person/mihai-surdeanu")
  )
)
ThisBuild / homepage := Some(url(s"https://github.com/clulab/$publication"))
ThisBuild / licenses := List(
  "Apache License, Version 2.0" ->
      url("http://www.apache.org/licenses/LICENSE-2.0.html")
)
ThisBuild / organization := "org.clulab"
ThisBuild / organizationHomepage := Some(url("http://clulab.org/"))
ThisBuild / organizationName := "Computational Language Understanding (CLU) Lab"
// The sonatype plugin seems to overwrite these two values.
// ThisBuild / pomIncludeRepository := { _ => false } // no longer applicable in sbt 1.6.2+
ThisBuild / pomPostProcess := {
  val logback = DependencyId("ch.qos.logback", "logback-classic")
  val rule = DependencyFilter { dependencyId =>
    dependencyId != logback
  }

  (node: Node) => new RuleTransformer(rule).transform(node).head
}
// ThisBuild / publishMavenStyle := true // no longer applicable in sbt 1.6.2+
ThisBuild / publishTo := {
  if (BuildUtils.artifactory) {
    val realm = "Artifactory Realm"
    val provider = "https://artifactory.clulab.org/artifactory/"
    val repository = "sbt-release-local"
    val details =
        if (isSnapshot.value) ";build.timestamp=" + new java.util.Date().getTime
        else ""
    val location = provider + repository + details

    Some(realm at location)
  }
  else {
    val realm = if (isSnapshot.value) "snapshots" else "releases"
    val provider = "https://oss.sonatype.org/"
    val repository = ""
    val details =
        if (isSnapshot.value) "content/repositories/snapshots"
        else "service/local/staging/deploy/maven2"
    val location = provider + repository + details

    Some(realm at location)
  }
}
ThisBuild / scmInfo := Some(
  ScmInfo(
    url(s"https://github.com/clulab/$publication"),
    s"scm:git@github.com:clulab/$publication.git"
  )
)
