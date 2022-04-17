#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

class BCLoaderNew : llvm::PassInfoMixin<BCLoaderNew>{
  public:
    llvm::PreservedAnalyses run(llvm::Module & M, 
                                llvm::ModuleAnalysisManager &MAM);
};

llvm::ModulePass *createBCLoaderPass();
