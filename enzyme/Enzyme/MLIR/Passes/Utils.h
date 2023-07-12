//===- Utils.h - General Utilities -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Dialect/Ops.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
namespace mlir {
namespace enzyme {

class Utils {
public:
  static mlir::linalg::GenericOp adjointToGeneric(enzyme::GenericAdjointOp &op,
                                                  OpBuilder &builder,
                                                  Location loc);
};

} // namespace enzyme
} // namespace mlir