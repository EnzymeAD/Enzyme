//===- EnzymeMLIRPasses.h - C API for Enzyme MLIR passes ----------*- C -*-===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares a C constructor and registration function for every
// Enzyme MLIR pass, generated from Passes.td: `mlirCreateEnzyme<Pass>`
// creates the pass with its default options, `mlirRegisterEnzyme<Pass>`
// registers it, and `mlirRegisterEnzymePasses` registers them all.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIRPASSES_H_
#define ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIRPASSES_H_

#include "mlir-c/Pass.h"

#include "Passes/Passes.capi.h.inc"

#endif // ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIRPASSES_H_
