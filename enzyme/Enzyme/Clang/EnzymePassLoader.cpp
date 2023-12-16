//===- EnzymePassLoader.cpp - Automatic Differentiation Transformation
// Pass---===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains a clang plugin for Enzyme.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/llvm-config.h"

#if LLVM_VERSION_MAJOR < 16

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include "../Enzyme.h"
#include "../PreserveNVVM.h"

extern llvm::cl::opt<bool> EnzymeEnable;

using namespace llvm;

// This function is of type PassManagerBuilder::ExtensionFn
static void loadPass(const PassManagerBuilder &Builder,
                     legacy::PassManagerBase &PM) {
  if (!EnzymeEnable)
    return;
  PM.add(createPreserveNVVMPass(/*Begin=*/true));
  PM.add(createGVNPass());
  PM.add(createSROAPass());
  PM.add(createEnzymePass(/*PostOpt*/ true));
  PM.add(createPreserveNVVMPass(/*Begin=*/false));
  PM.add(createGVNPass());
  PM.add(createSROAPass());
  PM.add(createLoopDeletionPass());
  PM.add(createGlobalOptimizerPass());
  // PM.add(SimplifyCFGPass());
}

static void loadNVVMPass(const PassManagerBuilder &Builder,
                         legacy::PassManagerBase &PM) {
  PM.add(createPreserveNVVMPass(/*Begin=*/true));
}

// These constructors add our pass to a list of global extensions.
static RegisterStandardPasses
    clangtoolLoader_Ox(PassManagerBuilder::EP_VectorizerStart, loadPass);
static RegisterStandardPasses
    clangtoolLoader_O0(PassManagerBuilder::EP_EnabledOnOptLevel0, loadPass);
static RegisterStandardPasses
    clangtoolLoader_OEarly(PassManagerBuilder::EP_EarlyAsPossible,
                           loadNVVMPass);

static void loadLTOPass(const PassManagerBuilder &Builder,
                        legacy::PassManagerBase &PM) {
  if (!EnzymeEnable)
    return;
  loadPass(Builder, PM);
  PassManagerBuilder Builder2 = Builder;
  Builder2.Inliner = nullptr;
  Builder2.LibraryInfo = nullptr;
  Builder2.ExportSummary = nullptr;
  Builder2.ImportSummary = nullptr;
  /*
  Builder2.LoopVectorize = false;
  Builder2.SLPVectorize = false;
  Builder2.DisableUnrollLoops = true;
  Builder2.RerollLoops = true;
  */
  const_cast<PassManagerBuilder &>(Builder2).populateModulePassManager(PM);
}

static RegisterStandardPasses
    clangtoolLoader_LTO(PassManagerBuilder::EP_FullLinkTimeOptimizationEarly,
                        loadLTOPass);

#endif // LLVM_VERSION_MAJOR < 16
