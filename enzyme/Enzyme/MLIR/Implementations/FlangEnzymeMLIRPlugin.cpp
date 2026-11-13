//===- FlangEnzymeMLIRPlugin.cpp - Enzyme's plugin for flang's MLIR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FlangEnzymeMLIR is loaded into `flang -fc1`, with -load or by the LLVM pass
// plugin FlangEnzyme (see FlangEnzymeMLIRLoader.cpp). Static initializers in
// it add Enzyme's passes to flang's MLIR pipeline through
// fir::registerPassPipelineConfigCallback.
//
// Only the Enzyme code is carried here; MLIR, FIR and flang symbols resolve
// from the host at load time.
//
//===----------------------------------------------------------------------===//
