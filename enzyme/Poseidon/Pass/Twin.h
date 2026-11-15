//===- Twin.h - the module as written, compiled beside the rewritten one --===//
// The twin is compiled as a build without a profile would compile it, and
// takes from the rewritten module only the functions a rewrite reached.
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

bool compileAsWritten(llvm::Module &twin, const Session &parent,
                      llvm::PassBuilder &PB, llvm::OptimizationLevel Level,
                      llvm::ThinOrFullLTOPhase Phase);
bool takeRewritten(llvm::Module &T, llvm::ArrayRef<llvm::Function *> rewritten);
void adoptModule(llvm::Module &M, llvm::Module &from);

} // namespace poseidon

#endif // POSEIDON_TWIN_H
