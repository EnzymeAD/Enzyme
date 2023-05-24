//===- TypeAnalysisPrinter.h - Printer utility pass for Type Analysis -----===//
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
// This file contains a utility LLVM pass for printing derived Type Analysis
// results of a given function.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"

namespace llvm {
class FunctionPass;
}

class TypeAnalysisPrinterNewPM final
    : public llvm::AnalysisInfoMixin<TypeAnalysisPrinterNewPM> {
  friend struct llvm::AnalysisInfoMixin<TypeAnalysisPrinterNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  TypeAnalysisPrinterNewPM() {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};
