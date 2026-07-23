//===- enzyme-fir-plugin.cpp - Enzyme as an MLIR plugin for fir-opt --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes Enzyme-MLIR as a pair of MLIR plugins that can be loaded
// into any `MlirOptMain`-based tool -- in particular Flang's `fir-opt`:
//
//   flang-new -fc1 -emit-hlfir foo.f90 -o foo.hlfir
//   fir-opt --load-dialect-plugin=FIREnzyme.so \
//           --load-pass-plugin=FIREnzyme.so \
//           -pass-pipeline='builtin.module(enzyme)' foo.hlfir -o foo.diff.hlfir
//
// This is the bring-up / FileCheck path for differentiating Fortran array
// intrinsics *while they are still first-class `hlfir.*` ops* (see
// PLAN_flang_enzyme_mlir.md), with zero changes to Flang itself: fir-opt is
// already `MlirOptMain`-based, so `--load-dialect-plugin` /
// `--load-pass-plugin` work out of the box.
//
// The dialect plugin registers the Enzyme dialect and the autodiff interface
// external models for the upstream core dialects (arith, math, scf, cf, ...).
// The pass plugin registers every Enzyme MLIR pass (`enzyme`, `enzyme-wrap`,
// ...). fir-opt itself already registers the FIR/HLFIR dialects, so a module
// containing `hlfir.*` ops is parsed by the host and differentiated by the
// Enzyme pass loaded here.
//
// The two entry points below (`mlirGetDialectPluginInfo` /
// `mlirGetPassPluginInfo`) are the symbols `MlirOptMain` looks up in a loaded
// shared object; because both live in this one file, a single `FIREnzyme.so`
// serves both `--load-dialect-plugin` and `--load-pass-plugin`.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Config/llvm-config.h"

#include "Dialect/Dialect.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Passes/Passes.h"

using namespace mlir;

/// Populate `registry` with the Enzyme dialect and the autodiff external models
/// for the upstream dialects Enzyme differentiates through. Shared by the
/// dialect-plugin entry point below.
static void registerEnzymeInto(DialectRegistry &registry) {
  registry.insert<mlir::enzyme::EnzymeDialect>();
  enzyme::registerCoreDialectAutodiffInterfaces(registry);
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Enzyme", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) { registerEnzymeInto(*registry); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Enzyme", LLVM_VERSION_STRING,
          []() { mlir::enzyme::registerenzymePasses(); }};
}
