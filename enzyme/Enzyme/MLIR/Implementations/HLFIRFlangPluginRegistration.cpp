//===- HLFIRFlangPluginRegistration.cpp - flang -load bridge --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// When this object is loaded into `flang -fc1` via -load, the static
// initializer below registers a callback with flang
// (fir::registerPassPipelineConfigCallback) that uses the HLFIROptEarly
// extension point to add two passes to flang's HLFIR-to-FIR pipeline, while the
// hlfir.* intrinsics are still present:
//
//   1. enzyme-lower-fortran-calls: f__enzyme_fwddiff/autodiff -> enzyme.* ops.
//   2. enzyme: differentiate those ops in place.
//
// Only the Enzyme code is carried here; MLIR/FIR/HLFIR and flang symbols
// resolve from the host at load time.
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

// Attach the Enzyme dialect and the Fortran autodiff models to the flang-owned
// context. flang has already loaded FIR/HLFIR/func/... into it, so appending
// the registry applies their extensions immediately and defers the rest.
void appendEnzymeFortranInterfaces(MLIRContext &context) {
  DialectRegistry registry;
  mlir::enzyme::registerEnzymeFortranInterfaces(registry);
  mlir::enzyme::registerFIRDialectAutoDiffInterface(registry);
  mlir::enzyme::registerHLFIRDialectAutoDiffInterface(registry);
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
                pm.addPass(mlir::enzyme::createHLFIRLowerEnzymeCallsPass());
                pm.addPass(mlir::enzyme::createDifferentiatePass());
              });
        });
  }
};
// Runs when `flang -fc1 -load` dlopens this object.
static EnzymeFlangPipelineRegistration enzymeFlangPipelineRegistration;
} // namespace
