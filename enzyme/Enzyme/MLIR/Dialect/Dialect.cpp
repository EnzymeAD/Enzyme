//===- EnzymeDialect.cpp - Enzyme dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "Ops.h"

using namespace mlir;
using namespace mlir::enzyme;

//===----------------------------------------------------------------------===//
// Enzyme dialect.
//===----------------------------------------------------------------------===//

void EnzymeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/EnzymeOps.cpp.inc"
      >();
}

#include "Dialect/EnzymeOpsDialect.cpp.inc"
