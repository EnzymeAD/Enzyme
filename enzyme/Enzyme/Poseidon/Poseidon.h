#ifndef ENZYME_POSEIDON_H
#define ENZYME_POSEIDON_H

#include <string>

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

extern "C" {
extern llvm::cl::opt<bool> FPProfileGenerate;
extern llvm::cl::opt<std::string> FPProfileUse;
extern llvm::cl::opt<bool> FPOptPrint;
extern llvm::cl::opt<bool> FPOptEnableHerbie;
extern llvm::cl::opt<bool> FPOptEnablePT;
extern llvm::cl::opt<bool> FPOptEnableSolver;
extern llvm::cl::opt<unsigned> FPOptMaxExprDepth;
extern llvm::cl::opt<unsigned> FPOptMaxExprLength;
extern llvm::cl::opt<std::string> FPOptReductionEval;
extern llvm::cl::opt<std::string> FPOptCachePath;
extern llvm::cl::opt<bool> FPOptMultiOutputPTOnly;
}

bool Poseidonable(const Value &V);
void setPoseidonMetadata(Function &F);
void preprocessForPoseidon(Function *F);
bool fpOptimize(Function &F, const TargetTransformInfo &TTI,
                double errorTol = 0.0);

class FPOpt final : public FunctionPass {
public:
  static char ID;
  FPOpt();

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};

llvm::FunctionPass *createFPOptPass();

class FPOptNewPM final : public llvm::AnalysisInfoMixin<FPOptNewPM> {
  friend struct llvm::AnalysisInfoMixin<FPOptNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  FPOptNewPM() {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

#endif // ENZYME_POSEIDON_H
