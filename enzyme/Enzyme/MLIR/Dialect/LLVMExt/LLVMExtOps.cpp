#include "Dialect/LLVMExt/LLVMExt.h"

namespace mlir {
namespace enzyme {} // namespace enzyme
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/LLVMExt/LLVMExtOps.cpp.inc"

using namespace mlir;
using namespace mlir::enzyme;

// A copy is lowered as one operation in one memory space -- a host memcpy or a
// device-to-device runtime copy -- so a mismatch here is a transfer nothing
// downstream can express, not something to silently pick a side of.
LogicalResult llvm_ext::MemcpyOp::verify() {
  auto dstSpace =
      cast<LLVM::LLVMPointerType>(getDst().getType()).getAddressSpace();
  auto srcSpace =
      cast<LLVM::LLVMPointerType>(getSrc().getType()).getAddressSpace();
  if (dstSpace != srcSpace)
    return emitOpError() << "destination is in memory space " << dstSpace
                         << " but source is in memory space " << srcSpace
                         << "; a copy across memory spaces is not supported";
  return success();
}
