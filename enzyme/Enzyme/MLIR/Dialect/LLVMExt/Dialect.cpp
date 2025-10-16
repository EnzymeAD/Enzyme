#include "Dialect.h"

#include "Enzyme/MLIR/Dialect/LLVMExt/LLVMExtDialect.cpp.inc"

// Initialize the dialect
void mlir::enzyme::llvm_ext::LLVMExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Enzyme/MLIR/Dialect/LLVMExt/Ops.cpp.inc"
      >();
}
