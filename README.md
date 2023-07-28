[![Build Status](https://github.com/clulab/scala-transformers/workflows/Scala-Transformers%20CI/badge.svg)](https://github.com/clulab/scala-transformers/actions)
[![Maven Central](https://img.shields.io/maven-central/v/org.clulab/scala-transformers-tokenizer_2.12)](https://mvnrepository.com/artifact/org.clulab/scala-transformers-tokenizer)


# scala-transformers
Scala interfaces to newly trained [Hugging Face](https://huggingface.co/)/[ONNX](https://onnx.ai/) transformers and existing tokenizers

The libraries and models resulting from this project are incorporated into [processors](https://github.com/clulab/processors) and generally don't need attention unless functionality is being modified, but here are some details about how it all works.


## encoder

To incorporate the encoder subproject as a Scala library dependency, either to access an existing model or because you've trained a new one with the Python code there, you'll need to add something like this to your `build.sbt` file:

```scala
libraryDependencies += "org.clulab" %% "scala-transformers-encoder" % "0.3.0"
```

New models should generally be published to the [CLU Lab's artifactory server](https://artifactory.clulab.org/) so that they can be treated as library dependencies, although they can also be accessed as local files.  Two models have been generated and published.  They are incorporated into a Scala project with

```scala
resolvers += "clulab" at "https://artifactory.clulab.org/artifactory/sbt-release"

// Pick one or more.
libraryDependencies += "org.clulab" % "deberta-onnx-model"  % "0.0.3"
libraryDependencies += "org.clulab" % "roberta-onnx-model"  % "0.0.2"

```

The models make reference to tokenizers which also need to be added according to instructions in the next section.

Please see the [encoder README](encoder/README.md) for information about how to generate models and how to download and package Hugging Face tokenizers for use in the tokenizer subproject.

## tokenizer

To use the tokenizer subproject as a Scala library dependency, you'll need to add something like this to your `build.sbt` file:

```scala
libraryDependencies += "org.clulab" %% "scala-transformers-tokenizer" % "0.3.0"
```

See the [tokenizer README](tokenizer/README.md) for information about which tokenizers have already been packaged and how they are accessed.
