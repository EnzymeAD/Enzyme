#ifndef ENZYME_POSEIDON_H
#define ENZYME_POSEIDON_H

#include <string>

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

extern "C" {
extern llvm::cl::opt<bool> EnzymeEnableFPOpt;
extern llvm::cl::opt<bool> EnzymePrintFPOpt;
extern llvm::cl::opt<bool> FPOptPrintPreproc;
}

extern llvm::cl::opt<std::string> FPOptTargetFuncRegex;
extern llvm::cl::opt<bool> FPOptEnableHerbie;
extern llvm::cl::opt<bool> FPOptEnablePT;
extern llvm::cl::opt<bool> FPOptEnableSolver;
extern llvm::cl::opt<unsigned> FPOptMaxFPCCDepth;
extern llvm::cl::opt<unsigned> FPOptMaxExprDepth;
extern llvm::cl::opt<unsigned> FPOptMaxExprLength;
extern llvm::cl::opt<std::string> FPOptReductionProf;
extern llvm::cl::opt<std::string> FPOptReductionEval;
extern llvm::cl::opt<std::string> FPOptCachePath;

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
