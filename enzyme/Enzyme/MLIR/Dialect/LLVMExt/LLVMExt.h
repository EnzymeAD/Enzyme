//===- LLVMExt.h - LLVMExt dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_LLVMEXT_H
#define ENZYME_LLVMEXT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// Include the dialect
#include "Dialect/LLVMExt/LLVMExtOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/LLVMExt/LLVMExtOps.h.inc"

#endif // ENZYME_LLVMEXT_H
