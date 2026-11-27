//=- TypeHintsToValues.h - Move type hints off promotable allocas =========//
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
// This file declares TypeHintsToValues, which replaces the __enzyme_float,
// __enzyme_double, __enzyme_integer and __enzyme_pointer hints on an alloca
// that does not otherwise escape by type annotations on the values loaded from
// the hinted memory, so that the hints no longer keep SROA from promoting the
// alloca.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_TYPE_HINTS_TO_VALUES_H
#define ENZYME_TYPE_HINTS_TO_VALUES_H

#include <llvm/Config/llvm-config.h>

#include "PassUtils.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
}

class TypeHintsToValuesNewPM final : public PassParent<TypeHintsToValuesNewPM> {
  friend PassParent<TypeHintsToValuesNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  TypeHintsToValuesNewPM() {}

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  static bool isRequired() { return true; }
};

/// Replace every __enzyme_typed_value call that TypeHintsToValues created by
/// its operand, moving the call's enzyme_type annotation onto the operand when
/// it is an instruction. Enzyme calls this before analyzing any function.
bool lowerTypedValues(llvm::Module &M);

#endif // ENZYME_TYPE_HINTS_TO_VALUES_H
