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

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
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

// The Enzyme dialect and the autodiff interface external models the `enzyme`
// differentiation pass needs to differentiate HLFIR/FIR + the upstream dialects
// that Fortran lowers to (arith, math, complex, cf, scf, tensor, memref, func,
// LLVM). Deliberately a *subset* of registerCoreDialectAutodiffInterfaces: the
// Linalg models are omitted because Fortran HLFIR never produces linalg ops at
// this stage and -- unlike every other dialect here -- flang does not export
// the Linalg dialect symbols for the lean plugin to resolve against. Affine,
// NVVM and LLVMExt are likewise unnecessary for the HLFIR path.
void appendEnzymeFortranInterfaces(MLIRContext &context) {
  DialectRegistry registry;
  registry.insert<mlir::enzyme::EnzymeDialect>();
  mlir::enzyme::registerArithDialectAutoDiffInterface(registry);
  mlir::enzyme::registerBuiltinDialectAutoDiffInterface(registry);
  mlir::enzyme::registerComplexDialectAutoDiffInterface(registry);
  mlir::enzyme::registerLLVMDialectAutoDiffInterface(registry);
  mlir::enzyme::registerMathDialectAutoDiffInterface(registry);
  // NB: the MemRef autodiff model is intentionally omitted -- Fortran lowers to
  // !fir.ref, not memref, and its zeroInPlace pulls in linalg.fill (a Linalg
  // symbol flang does not export). Add it back with a linalg-free zeroing path
  // if a memref-caching reverse-mode configuration ever needs it. The LLVM model
  // is kept (it references Enzyme's own llvm_ext dialect, which the plugin
  // bundles) so the pass can also handle any llvm.* that reaches it.
  mlir::enzyme::registerSCFDialectAutoDiffInterface(registry);
  mlir::enzyme::registerCFDialectAutoDiffInterface(registry);
  mlir::enzyme::registerFuncDialectAutoDiffInterface(registry);
  mlir::enzyme::registerTensorDialectAutoDiffInterface(registry);
  mlir::enzyme::registerEnzymeDialectAutoDiffInterface(registry);
  // !fir.ref active-memory models (by-reference Fortran scalars) and the HLFIR
  // intrinsics (hlfir.matmul, ...).
  mlir::enzyme::registerFIRDialectAutoDiffInterface(registry);
  mlir::enzyme::registerHLFIRDialectAutoDiffInterface(registry);

  // flang has already created and loaded the FIR/HLFIR/func/... dialects into
  // this context; appending applies the external-model extensions to them
  // immediately and defers the rest until their dialect loads.
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
