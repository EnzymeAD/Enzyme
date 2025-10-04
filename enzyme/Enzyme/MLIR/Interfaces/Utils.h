//===- OperationUtils.h - Utilities for gradient interfaces -------* C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/Operation.h"

namespace mlir {
namespace enzyme {
namespace oputils {
// Checks if the operation/function has any memory write effects. This enables
// batching specific AD optimiziations(which are triggered only if the primal
// function doesnt modify memory operands)
bool isReadOnly(Operation *op);

bool mayAlias(Value v1, Value v2);
} // namespace oputils
} // namespace enzyme
} // namespace mlir
