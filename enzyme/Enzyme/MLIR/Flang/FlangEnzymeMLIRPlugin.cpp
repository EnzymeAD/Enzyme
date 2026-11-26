//===- FlangEnzymeMLIRPlugin.cpp - Enzyme's plugin for flang's MLIR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The plugin through which Enzyme adds passes to flang's MLIR pipeline. It is
// loaded with `flang -fc1 -load` or by FlangEnzyme, and resolves MLIR, FIR and
// flang symbols from the host. It registers no passes yet.
//
//===----------------------------------------------------------------------===//
