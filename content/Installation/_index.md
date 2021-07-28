---
title: "Installation"
date: 2019-11-29T15:26:15Z
draft: false
weight: 5
---

You can install Enzyme using [Homebrew](https://brew.sh), or build it manually from source. To install using Homebrew, run

```
brew install enzyme
```

Currently, Homebrew has a pre-built binary (bottle) on macOS, but will build from source on Linux. To request a bottle for Linux, [file an issue](https://github.com/Homebrew/linuxbrew-core/issues) at Homebrew.

The rest of these instructions explain how to build Enzyme from source.

## Downloading Enzyme
To start you should download Enzyme's code [Github](https://github.com/wsmoses/Enzyme).

```sh
git clone https://github.com/wsmoses/Enzyme
cd Enzyme
```


## Building LLVM

Enzyme is a plugin for LLVM and consequently needs an existing build of LLVM to function.

Enzyme is designed to work with a wide range of LLVM versions and is currently tested against LLVM 7, 8, 9, 10, 11, 12, and mainline. LLVM's plugin infrastructure can sometimes be flakey or not built by default. If loading Enzyme into an existing LLVM installation results in segfaults, we recommend building LLVM from source.

Details on building LLVM can be found in for building LLVM can be found in the [LLVM Getting Started](https://llvm.org/docs/GettingStarted.html). A simple build command using Ninja is shown below:

```sh
cd /path/to/llvm/source/
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_PLUGINS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
ninja
```

## Building Enzyme

First enter the enzyme project directory inside the monorepo and create a build directory. Note that the source is inside a subdirectory of the repo called enzyme. If you get a CMakeLists.txt not found error, make sure you're pointing at the enzyme subdirectory.

```sh
cd /path/to/Enzyme/enzyme
mkdir build && cd build
```

From here, we can configure Enzyme via cmake, then build. Again, for ease we use Ninja (requiring the ninja build to to be installed). One can use make by omitting `-G Ninja` and running make instead of ninja.

```sh
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
ninja
```

This should create a file `Enzyme/LLVMEnzyme-<VERSION>.so` inside the build directory, which contains the LLVM analysis and optimization passes that implement Enzyme.



## Verifying installation

We can run Enzyme's unit tests by running the following command. They should run in less than a minute and verify that your build of Enzyme and LLVM interoperate properly.

```sh
ninja check-enzyme
```

We can also run Enzyme's C/C++ integration tests. These tests require an existing installation of the Adept AD Engine (to compare against) and Eigen. Running these tests will take a moderate amount of time (about 6 minutes on a recent multicore).

```sh
ninja check-enzyme-integration
```

Finally, we can also run Enzyme's benchmarking suite, which is composed of the reverse mode tests from Microsoft's [ADBench suite](https://github.com/microsoft/ADBench) and other interesting cases. Running these tests will potentially take a long time (about an hour on a recent multicore).

```sh
ninja bench-enzyme
````

If you run Enzyme tests and get an error like `/bin/sh: 1: ../../: Permission denied` or ` ../../ not found`, it's likely that cmake wasn't able to find your version of llvm-lit, LLVM's unit tester. This often happens if you use the default Ubuntu install of LLVM as they stopped including it in their packaging. To remedy, find lit.py or lit or llvm-lit on your system and add the following flag to cmake:
```sh
cmake .. -DLLVM_EXTERNAL_LIT=/path/to/lit/lit.py
```

## Developing inside a Container

For debugging and testing purposes we have created [Docker images](https://github.com/tgymnich/enzyme-dev-docker), which closely resemble all of our CI environments. If you are using Visual Studio Code you can build and test Enzyme inside of a [Dev Container](https://code.visualstudio.com/docs/remote/containers). To change either the Ubuntu or the LLVM version in Visual Studio Code just edit the file at `.devcontainer/devcontainer.json` accordingly.