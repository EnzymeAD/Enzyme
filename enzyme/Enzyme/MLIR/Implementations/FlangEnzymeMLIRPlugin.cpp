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
// fir::registerPassPipelineConfigCallback: at the FIROptLast extension point,
// while FIR still has the Fortran types, the enzyme-fir-type-annotations pass
// (-enzyme-fir-type-annotations=false turns it off), whose types LLVM
// Enzyme's type analysis reads where Enzyme runs on LLVM IR (e.g. in the
// link).
//
// Only the Enzyme code is carried here; MLIR, FIR and flang symbols resolve
// from the host at load time.
//
//===----------------------------------------------------------------------===//

#include "Implementations/FIRTypeAnnotations.h"

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
