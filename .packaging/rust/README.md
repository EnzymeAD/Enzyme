# Enzyme build helper

## Goal

This repository will build enzyme/llvm/clang/rustc in the right configuration such that you can use it in combination with [oxide-enzyme](https://github.com/rust-ml/oxide-enzyme).

## Requirements

 - git  
 - ninja  
 - cmake  
 - libssl-dev
 - libclang-dev
 - Rust (rustup) with an installed nightly toolchain   
 - ~10GB free storage in $HOME/.cache

## Usage

Build LLVM, the Rust toolchain and Enzyme with

```bash
cargo install enzyme
enzyme-install
```

Depending on your CPU this might take a few hours.  
The build process will run enzyme tests, so your last output should look similar to these lines:

Testing Time: 0.63s  
  Passed           : 299  
  Expectedly Failed:   5  

## Extras
- Q: It fails some (all) tests or the build breaks even earlier. Help?
- A: Sorry. Please open an issue with relevant information (OS, error message) or ping me on the rust-ml Zulip.  
&nbsp;
- Q: How often do I have to run this? It takes quite a while..
- A: We are aware of this and working on offering pre-build versions. In the meantime you have to build it once per rust Version. So we will publish an update once 1.58 hits stable.

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.
