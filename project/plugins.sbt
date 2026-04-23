// Latest version numbers were updated on 2021 Mar 11.
addSbtPlugin("com.jsuereth"      % "sbt-pgp"      % "1.1.2-1") // up to 1.1.2-1 *
// Deprecation Notice: This plugin no longer works as Sonatype has deprecated the legacy API
// Please use sbt's native Sonatype support instead. See the official sbt documentation for details.
// addSbtPlugin("org.xerial.sbt"    % "sbt-sonatype" % "2.3")     // up to 3.12.2 *
addSbtPlugin("com.github.sbt"    % "sbt-release"  % "1.4.0")  // up to 1.4.0
addSbtPlugin("io.get-coursier"   % "sbt-shading"  % "2.1.3")   // up to 2.1.3
// * Held back out of an abundance of caution.
