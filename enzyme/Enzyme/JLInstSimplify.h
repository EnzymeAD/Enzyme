//=- JLInstSimplify.h - Additional instsimplifyrules for julia programs =//
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

#ifndef ENZYME_JL_INST_SIMPLIFY_H
#define ENZYME_JL_INST_SIMPLIFY_H

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/PassManager.h"

namespace llvm {
class FunctionPass;
}

class JLInstSimplifyNewPM final
    : public llvm::AnalysisInfoMixin<JLInstSimplifyNewPM> {
  friend struct llvm::AnalysisInfoMixin<JLInstSimplifyNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  JLInstSimplifyNewPM() {}

  Result run(llvm::Function &M, llvm::FunctionAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

#endif // ENZYME_JL_INST_SIMPLIFY_H
