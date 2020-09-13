---
title: "Users of MLIR"
date: 2019-11-29T15:26:15Z
draft: false
weight: 1
---

In alphabetical order below.

## [CIRCT](https://github.com/llvm/circt): Circuit IR Compilers and Tools

The CIRCT project is an (experimental!) effort looking to apply MLIR and the LLVM
development methodology to the domain of hardware design tools.

## [Flang](https://github.com/llvm/llvm-project/tree/master/flang)

Flang is a ground-up implementation of a Fortran front end written in modern C++.
It started off as the [f18 project](https://github.com/flang-compiler/f18) with an
aim to replace the previous [flang project](https://github.com/flang-compiler/flang)
and address its various deficiencies. F18 was subsequently accepted into the LLVM
project and rechristened as Flang. The high level IR of the Fortran compiler is modeled
using MLIR.

## [IREE](https://github.com/google/iree)

IREE (pronounced "eerie") is a compiler and minimal runtime system for
compiling ML models for execution against a HAL (Hardware Abstraction Layer)
that is aligned with Vulkan. It aims to be a viable way to compile and run
ML devices on a variety of small and medium sized systems, leveraging either
the GPU (via Vulkan/SPIR-V), CPU or some combination. It also aims to
interoperate seamlessly with existing users of Vulkan APIs, specifically
focused on games and rendering pipelines.

## [NPComp](https://github.com/llvm/mlir-npcomp): MLIR based compiler toolkit for numerical python programs

The NPComp project aims to provide tooling for compiling numerical python programs of various forms to take advantage of MLIR+LLVM code generation and backend runtime systems.

In addition to providing a bridge to a variety of Python based numerical programming frameworks, NPComp also directly develops components for tracing and compilation of generic Python program fragments.

## [ONNX-MLIR](https://github.com/onnx/onnx-mlir)

To represent neural network models, users often use [Open Neural Network
Exchange (ONNX)](http://onnx.ai/onnx-mlir/) which is an open standard format for
machine learning interoperability.
ONNX-MLIR is a MLIR-based compiler for rewriting a model in ONNX into a standalone
binary that is executable on different target hardwares such as x86 machines,
IBM Power Systems, and IBM System Z.

See also this paper: [Compiling ONNX Neural Network Models Using
MLIR](https://arxiv.org/abs/2008.08272).

## [PlaidML](https://github.com/plaidml/plaidml)

PlaidML is a tensor compiler that facilitates reusable and performance portable
ML models across various hardware targets including CPUs, GPUs, and
accelerators.

## [RISE](https://rise-lang.org/)

RISE is a spiritual successor to the
[Lift project](http://www.lift-project.org/): "a high-level functional data
parallel language with a system of rewrite rules which encode algorithmic
and hardware-specific optimisation choices".

## [TRFT: TensorFlow Runtime](https://github.com/tensorflow/runtime)

TFRT aims to provide a unified, extensible infrastructure layer for an
asynchronous runtime system.

## [TensorFlow](https://www.tensorflow.org/mlir)

MLIR is used as a Graph Transformation framework and the foundation for
building many tools (XLA, TFLite converter, quantization, ...).

## [Verona](https://github.com/microsoft/verona)

Project Verona is a research programming language to explore the concept of
concurrent ownership. They are providing a new concurrency model that seamlessly
integrates ownership.
