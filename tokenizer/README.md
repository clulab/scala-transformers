# tokenizer

## Scala

This subproject (the one in this tokenizer directory) should work on the four supported platforms without further measure (such as installation of Rust).  A build file is included for `sbt`, and `IntelliJ` is able to import it.  The pre-compiled library files in the [resources directory](./src/main/resources) will be used:

* apple-librust_tokenizer.jnilib is for Macs with Apple processors
* intel-librust_tokenizer.jnilib is for Macs with Intel processors
* librust_tokenizer.so is built for Linux with Intel processors
* rust_tokenizer.dll works under Windows for Intel processors

Additionally, the [tokenizer subdirectory](./src/main/resources/org/clulab/scala_transformers/tokenizer) includes a collection of pretrained Hugging Face tokenizers in serialized form.  Some of the tokenizers are available from the [Hugging Face website](https://huggingface.co/), but including them here allows the library to function without a network connection.  Other tokenizers are not directly available from the website.  These have been downloaded in their Python representation, converted to Rust format, and added to the resources.  In this way the library is also independent of Python at runtime.  The currently included tokenizers are

* [bert-base-cased](https://huggingface.co/bert-base-cased)
* [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)
* [roberta-base](https://huggingface.co/roberta-base)
* [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
* [google/bert_uncased_L-4_H-512_A-8](https://huggingface.co/google/bert_uncased_L-4_H-512_A-8)
* [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)
* [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)

For instructions on adding more, see the [encoder README](../encoder/README.md).

## Rust

To rebuild the libraries which provide the JNI interface to Hugging Face tokenizers, install `Rust` and then run `cargo build` in the [rust subdirectory](./src/main/rust).  Copy the resulting library (i.e., .so, .dll, or .dylib file) from the `target` directory to the [resources directory](./src/main/resources).  For a release, use `cargo build --release`.  The Macintosh library files will need to be renamed to distinguish between microprocessors, and the `.dylib` extension should be changed to `.jnilib`.
