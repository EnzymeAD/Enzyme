//=- Twin.h - the module as written, compiled beside the rewritten one ------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The twin is compiled as a build without a profile would compile it, and
// takes from the rewritten module only the functions a rewrite reached.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TWIN_H
#define POSEIDON_TWIN_H

#include "Session.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace poseidon {

// The analysis managers the plugin's nested pipelines run against. They are
// registered through the surrounding pipeline's PassBuilder, so that another
// plugin's pipeline names resolve, and kept local, so that nothing the outer
// pipeline cached reaches a pass that runs out of order. The session of the
// module is registered first, so that it is the one they hand out.
struct LocalAnalysisManagers {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  LocalAnalysisManagers(llvm::PassBuilder &PB, Session &S) {
    MAM.registerPass([&S] { return SessionAnalysis(S); });
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }
};

// Runs the default pipeline of `Phase` on `twin` as a build without a profile,
// the diagnostics already given for the module's sites silenced. False if it
// reported an error.
bool compileAsWritten(llvm::Module &twin, const Session &parent,
                      llvm::PassBuilder &PB, llvm::OptimizationLevel Level,
                      llvm::ThinOrFullLTOPhase Phase);
// Into `T`, the module as written compiled, the functions in `rewritten`, each
// in place of T's of the same name, with what they refer to. False if a
// reference has no counterpart in T, which is then unusable.
bool takeRewritten(llvm::Module &T, llvm::ArrayRef<llvm::Function *> rewritten);
// The contents of `M` replaced by those of `from`, which is left empty.
void adoptModule(llvm::Module &M, llvm::Module &from);

} // namespace poseidon

#endif // POSEIDON_TWIN_H
