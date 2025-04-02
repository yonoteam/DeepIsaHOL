name := "Isabelle-RL"

scalaVersion := "2.13.14"


lazy val root = project
  .in(file("."))
  .settings(
    name := "isabelle-rl",
    version := "0.1.0-SNAPSHOT",

    libraryDependencies += "de.unruh" %% "scala-isabelle" % "master-SNAPSHOT", // do `sbt publishLocal` in scala-isabelle first
    resolvers ++= Resolver.sonatypeOssRepos("snapshots"),

    // https://mvnrepository.com/artifact/io.circe
    scalacOptions += "-deprecation",
    libraryDependencies ++= Seq(
      "io.circe" %% "circe-core"    % "0.14.1",
      "io.circe" %% "circe-generic" % "0.14.1",
      "io.circe" %% "circe-parser"  % "0.14.1"
      ),
    
    /* From scala-isabelle and PISA */
    // https://mvnrepository.com/artifact/org.log4s/log4s
    libraryDependencies += "org.log4s" %% "log4s" % "1.10.0",
    // https://mvnrepository.com/artifact/org.slf4j/slf4j-simple
    libraryDependencies += "org.slf4j" % "slf4j-simple" % "2.0.13",
    // https://mvnrepository.com/artifact/commons-io/commons-io
    libraryDependencies += "commons-io" % "commons-io" % "2.16.1",
    // https://mvnrepository.com/artifact/org.scalaz/scalaz-core
    libraryDependencies += "org.scalaz" %% "scalaz-core" % "7.3.8",
    // https://mvnrepository.com/artifact/org.apache.commons/commons-lang3
    libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.14.0",
    // https://mvnrepository.com/artifact/org.apache.commons/commons-text
    libraryDependencies += "org.apache.commons" % "commons-text" % "1.12.0",
    // https://mvnrepository.com/artifact/com.google.guava/guava
    libraryDependencies += "com.google.guava" % "guava" % "33.2.1-jre",
    libraryDependencies += "org.jetbrains" % "annotations" % "24.1.0",
    libraryDependencies += "com.ibm.icu" % "icu4j" % "75.1",
    // https://mvnrepository.com/artifact/net.sf.py4j/py4j
    libraryDependencies += "net.sf.py4j" % "py4j" % "0.10.9.7"
  )
