//===- HLFIRFlangPluginRegistration.cpp - flang -load bridge --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// When this object is loaded into `flang -fc1` via -load, the static
// initializer below registers a config-augmentor with flang
// (fir::registerPassPipelineConfigCallback). flang invokes it after building
// the codegen MLIRToLLVMPassPipelineConfig, and we use the HLFIROptEarly
// extension point to plug the Enzyme HLFIR passes into the HLFIR-to-FIR pass
// pipeline while the hlfir.* intrinsics are still present:
//
//   1. enzyme-lower-fortran-calls: f__enzyme_fwddiff/autodiff -> enzyme.* ops.
//   2. enzyme: differentiate those ops in place (forward / reverse).
//
// Base MLIR/FIR/HLFIR and the flang symbols are resolved from the host
// `flang -fc1` at load time; only the Enzyme-specific code is carried here.
//
//===----------------------------------------------------------------------===//

#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"
#include "Passes/Passes.h"

#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Tools/CrossToolHelpers.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

// Attach the Enzyme dialect + Fortran autodiff models (see
// registerEnzymeFortranInterfaces / EnzymeFortranAutoDiffRegistration.cpp) to
// this flang-owned context. flang has already created and loaded the
// FIR/HLFIR/func/... dialects into it, so appending the registry applies the
// external-model extensions to them immediately and defers the rest until their
// dialect loads.
void appendEnzymeFortranInterfaces(MLIRContext &context) {
  DialectRegistry registry;
  mlir::enzyme::registerEnzymeFortranInterfaces(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::enzyme::EnzymeDialect>();
}

struct EnzymeFlangPipelineRegistration {
  EnzymeFlangPipelineRegistration() {
    fir::registerPassPipelineConfigCallback(
        [](MLIRToLLVMPassPipelineConfig &config) {
          config.registerHLFIROptEarlyEPCallbacks(
              [](mlir::PassManager &pm, llvm::OptimizationLevel) {
                appendEnzymeFortranInterfaces(*pm.getContext());
                // Fortran hook calls -> enzyme.fwddiff/autodiff ops.
                pm.addPass(mlir::enzyme::createHLFIRLowerEnzymeCallsPass());
                // Differentiate the emitted enzyme.* ops in place.
                pm.addPass(mlir::enzyme::createDifferentiatePass());
              });
        });
  }
};
// Runs when the shared object is dlopen'd by `flang -fc1 -load`.
static EnzymeFlangPipelineRegistration enzymeFlangPipelineRegistration;
} // namespace
