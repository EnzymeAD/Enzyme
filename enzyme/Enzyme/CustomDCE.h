#ifndef ENZYME_CUSTOM_DCE_H
#define ENZYME_CUSTOM_DCE_H

#include "llvm/IR/PassManager.h"

class CustomDCEPass final : public llvm::PassInfoMixin<CustomDCEPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
};

#endif
