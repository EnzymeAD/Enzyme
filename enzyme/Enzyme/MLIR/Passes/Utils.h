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

bool blockCmp(mlir::Block *a, mlir::Block *b);

bool opCmp(mlir::Operation *a, mlir::Operation *b);

bool regionCmp(mlir::Region *a, mlir::Region *b);

// This function returns whether a < b
bool valueCmp(mlir::Value a, mlir::Value b);

Type getConcatType(Value val, int64_t width);

Value getConcatValue(OpBuilder &builder, Location loc, ArrayRef<Value> argList);

Value getExtractValue(OpBuilder &builder, Location loc, Type argTy, Value val,
                      int64_t index);

void computeAffineIndices(OpBuilder &builder, Location loc, AffineMap map,
                          ValueRange operands, SmallVectorImpl<Value> &indices);

} // namespace enzyme
} // namespace mlir
