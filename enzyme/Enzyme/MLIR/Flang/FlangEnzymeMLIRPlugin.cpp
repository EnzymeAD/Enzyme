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
// At FIROptLast, while FIR still has the Fortran types, it runs
// enzyme-fir-type-annotations (off with -enzyme-fir-type-annotations=false).
// With a flang that has plugin directives, it also handles the !DIR$ ENZYME
// directives.
//
//===----------------------------------------------------------------------===//

#include "Flang/FIRTypeAnnotations.h"
#ifdef ENZYME_FLANG_PLUGIN_DIRECTIVES
#include "Flang/FortranDirectives.h"
#endif

#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Tools/CrossToolHelpers.h"

#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"

namespace {

llvm::cl::opt<bool> typeAnnotations(
    "enzyme-fir-type-annotations", llvm::cl::init(true),
    llvm::cl::desc("Carry the Fortran types that LLVM IR erases to LLVM "
                   "Enzyme's type analysis (all kinds; each kind has its own "
                   "-enzyme-fir-{arg,common,runtime,literal,local}-types)"));

struct EnzymeFlangPipelineRegistration {
  EnzymeFlangPipelineRegistration() {
#ifdef ENZYME_FLANG_PLUGIN_DIRECTIVES
    mlir::enzyme::registerFlangEnzymeDirectives();
#endif
    fir::registerPassPipelineConfigCallback(
        [](MLIRToLLVMPassPipelineConfig &config) {
          config.registerFIROptLastEPCallbacks(
              [](mlir::PassManager &pm, llvm::OptimizationLevel) {
                if (typeAnnotations)
                  pm.addPass(mlir::enzyme::createFIRTypeAnnotationsPass());
              });
        });
  }
};
// Runs when flang loads this object.
static EnzymeFlangPipelineRegistration enzymeFlangPipelineRegistration;

} // namespace
