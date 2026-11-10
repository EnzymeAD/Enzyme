// An opt pass plugin for expansion_fast_flags.cpp: applies the Poseidon
// plugin's own expansion materializer to every FP64 instruction of each
// function named ex<n>_*, as an n-component FP32 expansion, with no candidate
// selection in between. Loaded after the Poseidon plugin, whose applyExpansion
// it calls.
#include "Expansion.h"
#include "Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#if LLVM_VERSION_MAJOR >= 22 && !defined(ENZYME_LLVM22_PRERELEASE)
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif

using namespace llvm;

namespace {
struct ApplyExpansionPass : poseidon::PassParent<ApplyExpansionPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    for (Function &F : M) {
      StringRef name = F.getName();
      if (F.isDeclaration() || !name.starts_with("ex") || name.size() < 4 ||
          name[3] != '_')
        continue;
      const unsigned n = name[2] - '0';
      SmallVector<Instruction *, 16> insts;
      SmallPtrSet<Instruction *, 16> all;
      for (Instruction &I : instructions(F))
        if (I.getType()->isDoubleTy()) {
          insts.push_back(&I);
          all.insert(&I);
        }
      DenseMap<Value *, Value *> restored;
      if (n == 2)
        poseidon::applyExpansion(insts, all, &restored);
      else
        poseidon::applyExpansion(n, insts, all, &restored);
    }
    return PreservedAnalyses::none();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ApplyExpansion", "0", [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name != "apply-expansion")
                    return false;
                  MPM.addPass(ApplyExpansionPass());
                  return true;
                });
          }};
}
