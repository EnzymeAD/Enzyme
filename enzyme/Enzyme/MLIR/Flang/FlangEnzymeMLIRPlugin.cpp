//===- FlangEnzymeMLIRPlugin.cpp - Enzyme's plugin for flang's MLIR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A plugin for `flang -fc1` that adds Enzyme's passes to flang's MLIR pipeline
// through fir::registerPassPipelineConfigCallback. It is loaded with `-load`
// or by FlangEnzyme. MLIR, FIR and flang symbols resolve from the host.
//
//===----------------------------------------------------------------------===//
