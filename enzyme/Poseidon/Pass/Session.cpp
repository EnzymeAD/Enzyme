//===- Session.cpp - the compile of one module ----------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Session.h"

using namespace llvm;

namespace poseidon {

Session::Session(Options O, PassBuilder &PB, bool IsTwin)
    : Opts(std::move(O)), PB(PB), IsTwin(IsTwin),
      CostModel(Opts.Model.CostModel) {}

AnalysisKey SessionAnalysis::Key;

SessionAnalysis::SessionAnalysis(Options O, PassBuilder &PB)
    : Opts(std::move(O)), PB(&PB) {}

SessionAnalysis::SessionAnalysis(Session &S) : Given(&S) {}

SessionAnalysis::Result SessionAnalysis::run(Module &,
                                             ModuleAnalysisManager &) {
  Result R;
  if (Given) {
    R.S = Given;
  } else {
    R.Owned = std::make_unique<Session>(*Opts, *PB);
    R.S = R.Owned.get();
  }
  return R;
}

Session &sessionOf(Module &M, ModuleAnalysisManager &MAM) {
  return *MAM.getResult<SessionAnalysis>(M).S;
}

} // namespace poseidon
