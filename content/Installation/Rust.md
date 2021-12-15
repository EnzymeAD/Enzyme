---
title: "Rust"
date: 2019-11-29T15:26:15Z
draft: false
weight: 30
---

Getting started in Rust is easy, if you are using Linux. Otherwise, you probably need to wait for our next implementation, which will support Windows and Mac too.  
We have published `enzyme` on [crates.io](https://crates.io/crates/enzyme).

So after getting the following dependencies:

- git
- ninja
- cmake
- libssl-dev
- libclang-dev
- Rust (rustup) and any nightly toolchain 
- ~10GB free storage in $HOME/.cache

you can run
```sh
cargo install enzyme
```

This should only take a minute and give you an enzyme-install binary which you can use to build rustc/llvm/enzyme and a bunch of extra things, which might come helpful.  Now you can start the build process by calling
```bash
enzyme-install
```
It will build everything inside of `$HOME/.cache`, so please have around 10GB available there.
Also take care, the build process takes around one hour on a recent 8 core cpu.

## Verifying installation

Calling `enzyme-install` will automatically run Enzyme's unit tests after finishing the build process, so you should see something like
```sh
Testing Time: 0.63s
Passed : 299
Expectedly Failed: 5
```

You can also run a few basic Rust tests:
```sh
git clone https://github.com/rust-ml/oxide-enzyme
cd oxide-enzyme/example
cargo enzyme
```
You will find your binary slightly more hidden than usual in `target/$TARGET/debug/example`.

## Using Enzyme

In order to play nicely with cargo, the Rust workflow differs somewhat from the [C++](https://enzyme.mit.edu/getting_started/UsingEnzyme/) using Enzyme section.
First, you need to specify in your `build.rs` file how which function should be differentiated. 
Afterwards, you can run `cargo enzyme` without additional arguments, to build your crate.
If you want to have some more controll, you can also modify the following build command, which `cargo enzyme` is using:
```sh
RUSTFLAGS="--emit=llvm-bc" cargo +enzyme -Z build-std rustc --target x86_64-unknown-linux-gnu -- --emit=llvm-bc -g -C opt-level=3 -Zno-link && RUSTFLAGS="--emit=llvm-bc" cargo +enzyme -Z build-std rustc --target x86_64-unknown-linux-gnu -- --emit=llvm-bc -g -C opt-level=3
```

Please be aware that Enzyme currently can't be used in dependencies. However you can use Enzyme to differentiate functions and types defined in your dependencies. 
We still have quite a few issues open and are lacking some documentation. So you might find [this issue](https://github.com/rust-ml/oxide-enzyme/issues/6) and 
[this C++ documentation](https://enzyme.mit.edu/getting_started/CallingConvention/#types) useful to understand the Rust frontend. 
If you have specific question please open an issue and if you have some time, we are always happy to review PR's which add documentation or examples.
