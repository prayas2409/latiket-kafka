name := "kafkaStreaming"

version := "0.1"

scalaVersion := "2.12.8"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.3"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"

// https://mvnrepository.com/artifact/org.apache.spark/spark-streaming
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.4.3"

// https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % "2.8.7"

// https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-core
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-core" % "2.8.7"

// https://mvnrepository.com/artifact/com.fasterxml.jackson.module/jackson-module-scala
dependencyOverrides += "com.fasterxml.jackson.module" % "jackson-module-scala_2.12" % "2.8.7"

// https://mvnrepository.com/artifact/com.typesafe.play/play-json
libraryDependencies += "com.typesafe.play" %% "play-json" % "2.7.4"

// https://mvnrepository.com/artifact/org.apache.spark/spark-streaming-kafka-0-10
libraryDependencies += "org.apache.spark" %% "spark-streaming-kafka-0-10" % "2.4.3"

