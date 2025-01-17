#ifndef ENZYME_HERBIE_H
#define ENZYME_HERBIE_H

#include <string>

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/Support/CommandLine.h"

namespace llvm {
class FunctionPass;
}

extern "C" {
extern llvm::cl::opt<bool> EnzymeEnableFPOpt;
extern llvm::cl::opt<bool> EnzymePrintFPOpt;
}

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

#endif // ENZYME_HERBIE_H
