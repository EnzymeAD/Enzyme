#include "Dialect.h"

namespace mlir {
namespace enzyme {} // namespace enzyme
} // namespace mlir

#define GET_OP_CLASSES
#include "Enzyme/MLIR/Dialect/LLVMExt/LLVMExtOps.cpp.inc"
