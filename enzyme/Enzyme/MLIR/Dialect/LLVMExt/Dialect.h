#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// Include the dialect
#include "Enzyme/MLIR/Dialect/LLVMExt/LLVMExtDialect.h.inc"

// Operations
#define GET_OP_CLASSES
#include "Enzyme/MLIR/Dialect/LLVMExt/Ops.h.inc"
