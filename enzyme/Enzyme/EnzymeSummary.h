//===- EnzymeSummary.h - Per-module facts for a thin-link Enzyme step -----===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See EnzymeSummary.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_SUMMARY_H
#define ENZYME_SUMMARY_H

#include "PassUtils.h"

#include "llvm/IR/PassManager.h"

class EnzymeSummaryNewPM final : public PassParent<EnzymeSummaryNewPM> {
  friend PassParent<EnzymeSummaryNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);
  static bool isRequired() { return true; }
};

#endif
