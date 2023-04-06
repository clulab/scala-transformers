# tokenizer

This subproject should work on the four supported platforms without further measures being taken.  The pre-built library files in the resources directory will be used:

* apple-librust_tokenizer.jnilib is for Macs with Apple processors
* intel-librust_tokenizer.jnilib is for Macs with Intel processors
* librust_tokenizer.so is built for Linux with Intel processors
* rust_tokenizer.dll works under Windows for Intel processors

To rebuild these libraries, install `Rust` and then run `cargo build` in the `rust` directory.  Copy the resulting library from the `target` directory to the `resources` directory.  For a release, use `cargo build --release`.
