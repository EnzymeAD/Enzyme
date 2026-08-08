//===- PreserveNVVM.h - Mark NVVM attributes for preservation.  -------===//
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
// This file contains createPreserveNVVM, a transformation pass that marks
// calls to __nv_* functions, marking them as noinline as implementing the llvm
// intrinsic.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_PRESERVE_NVVM_H
#define ENZYME_PRESERVE_NVVM_H

#include "PassUtils.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class ModulePass;
class FunctionPass;
} // namespace llvm

// `PromoteMathLinkage` controls whether the preserved libdevice/math
// definitions are also promoted to external linkage so they survive to
// Enzyme. A consumer whose pipeline keeps them alive its own way (Reactant's
// raising consumes the math calls itself) passes false and gets only the
// inlining toggle.
llvm::ModulePass *createPreserveNVVMPass(bool Begin,
                                         bool PromoteMathLinkage = true);
llvm::FunctionPass *createPreserveNVVMFnPass(bool Begin);

class PreserveNVVMNewPM final : public PassParent<PreserveNVVMNewPM> {
  friend PassParent<PreserveNVVMNewPM>;

private:
  bool Begin;
  bool PromoteMathLinkage;
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  PreserveNVVMNewPM(bool Begin, bool PromoteMathLinkage = true)
      : Begin(Begin), PromoteMathLinkage(PromoteMathLinkage) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

#endif // ENZYME_PRESERVE_NVVM_H
