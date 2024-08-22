name := "Isabelle-RL"

scalaVersion := "2.13.14"


lazy val root = project
  .in(file("."))
  .settings(
    name := "isabelle-rl",
    version := "0.1.0-SNAPSHOT",

    libraryDependencies += "de.unruh" %% "scala-isabelle" % "master-SNAPSHOT", // development snapshot
    resolvers ++= Resolver.sonatypeOssRepos("snapshots")
  )
