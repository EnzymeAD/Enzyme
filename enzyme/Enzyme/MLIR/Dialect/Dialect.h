//===- Dialect.h - Enzyme dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_DIALECT_H
#define ENZYME_DIALECT_H

#include "mlir/IR/Dialect.h"

#include "llvm/ADT/StringMap.h"

#include <functional>

namespace mlir {
class RewriterBase;

namespace enzyme {
class MGradientUtils;
} // namespace enzyme
} // namespace mlir

#include "Dialect/EnzymeOpsDialect.h.inc"

#endif // ENZYME_DIALECT_H
