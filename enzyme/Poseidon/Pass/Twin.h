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

// The analysis managers the plugin's own nested pipelines run against. They are
// registered through the PassBuilder the surrounding pipeline was built with,
// which is what makes another plugin's pipeline name (Enzyme's "enzyme")
// resolvable from here; keeping them local means nothing the outer pipeline
// cached is handed to a pass that runs out of order. The session of the module
// they run on is registered first, so that it is the one they hand out.
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
