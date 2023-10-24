// Adds readonly/writeonly/... attributes to a BLAS function
void attributeBLAS(BlasInfo blas, llvm::Function *F);
// Adds attributes to a few more non blas functions
void attributeTablegen(llvm::Function &F);

// Differentiates BLAS functions
bool handleBLAS(llvm::CallInst &call, llvm::Function *called,BlasInfo blas,const std::vector<bool> &overwritten_args);
