//===- enzyme-fir-plugin.cpp - Enzyme as an MLIR plugin for fir-opt -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enzyme-MLIR as a dialect+pass plugin for any `MlirOptMain`-based tool, in
// particular Flang's stock `fir-opt`, which needs no changes to load it:
//
//   flang -fc1 -emit-hlfir foo.f90 -o foo.hlfir
//   fir-opt --load-dialect-plugin=FIREnzyme.so \
//           --load-pass-plugin=FIREnzyme.so \
//           --pass-pipeline='builtin.module(enzyme)' \
//           foo.hlfir -o foo.diff.hlfir
//
// Both plugin entry points live in this one file, so a single FIREnzyme.so
// serves --load-dialect-plugin and --load-pass-plugin. fir-opt registers the
// FIR/HLFIR dialects itself, so it parses the module and the Enzyme pass
// loaded here differentiates it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Config/llvm-config.h"

#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"
#include "Passes/Passes.h"

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Enzyme", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            mlir::enzyme::registerEnzymeFortranInterfaces(*registry);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Enzyme", LLVM_VERSION_STRING, []() {
            mlir::enzyme::registerenzymePasses();
            mlir::enzyme::registerHLFIRLowerEnzymeCallsPass();
          }};
}
