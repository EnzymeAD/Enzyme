#include "Dialect/LLVMExt/LLVMExt.h"
#include "Dialect/LLVMExt/LLVMExtOpsDialect.cpp.inc"

// Initialize the dialect
void mlir::enzyme::llvm_ext::LLVMExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/LLVMExt/LLVMExtOps.cpp.inc"
      >();
}
