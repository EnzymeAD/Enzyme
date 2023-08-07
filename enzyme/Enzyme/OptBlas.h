//===- OptBlas.h - Mark NVVM attributes for preservation.  -------===//
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
// This file contains createOptimizeBlas,
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"

namespace llvm {
class ModulePass;
}

bool optimizeFncsWithBlas(llvm::Module &M);

llvm::ModulePass *createOptimizeBlasPass(bool Begin);

class OptimizeBlasNewPM final
    : public llvm::AnalysisInfoMixin<OptimizeBlasNewPM> {
  // friend struct llvm::AnalysisInfoMixin<OptimizeBlasNewPM>;

private:
  bool Begin;
  static llvm::AnalysisKey Key;

public:
  OptimizeBlasNewPM(bool Begin) : Begin(Begin) {}

  bool runOnModule(llvm::Module &M);

  static bool isRequired() { return true; }
};
