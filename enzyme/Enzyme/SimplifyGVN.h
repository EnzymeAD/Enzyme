//=- SimplifyGVN.h - GVN-like load forwarding optimization ==============//
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
// This file declares SimplifyGVN, a GVN-like optimization pass that forwards
// loads from noalias/nocapture arguments to their corresponding stores.
//
// This pass provides an alternative to LLVM's built-in GVN pass without the
// instruction/offset limit imposed by memdep analysis, allowing it to handle
// cases with large numbers of memory operations and offsets.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_SIMPLIFY_GVN_H
#define ENZYME_SIMPLIFY_GVN_H

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"

namespace llvm {
class FunctionPass;
}

class SimplifyGVNNewPM final
    : public llvm::AnalysisInfoMixin<SimplifyGVNNewPM> {
  friend struct llvm::AnalysisInfoMixin<SimplifyGVNNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  SimplifyGVNNewPM() {}

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  static bool isRequired() { return true; }
};

#endif // ENZYME_SIMPLIFY_GVN_H
