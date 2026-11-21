//===- AutoDiffControlFlowInterface.cpp - Control flow for reverse mode ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the cf and arith defaults of the dialect interface
// through which reverse mode builds the control flow of its reverse sweep.
//
//===----------------------------------------------------------------------===//

#include "AutoDiffControlFlowInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace mlir::enzyme;

Type AutoDiffControlFlowInterface::getBranchIndexType(
    MLIRContext *context) const {
  return IntegerType::get(context, 32);
}

Value AutoDiffControlFlowInterface::createBranchIndex(OpBuilder &builder,
                                                      Location loc,
                                                      int64_t value) const {
  return arith::ConstantIntOp::create(builder, loc, value, 32);
}

Operation *AutoDiffControlFlowInterface::createBranch(OpBuilder &builder,
                                                      Location loc,
                                                      Block *dest) const {
  return cf::BranchOp::create(builder, loc, dest);
}

Operation *AutoDiffControlFlowInterface::createSwitch(
    OpBuilder &builder, Location loc, Value flag, Block *defaultDest,
    ArrayRef<int64_t> caseValues, BlockRange caseDests) const {
  SmallVector<APInt> values;
  for (int64_t v : caseValues)
    values.push_back(APInt(32, v, /*isSigned=*/true));
  return cf::SwitchOp::create(builder, loc, flag, defaultDest, ValueRange(),
                              values, caseDests,
                              SmallVector<ValueRange>(values.size()));
}

Value AutoDiffControlFlowInterface::createSelectOnBranchIndex(
    OpBuilder &builder, Location loc, Value flag, int64_t value, Value ifTrue,
    Value ifFalse) const {
  Value isValue =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, flag,
                            createBranchIndex(builder, loc, value));
  return arith::SelectOp::create(builder, loc, isValue, ifTrue, ifFalse);
}

const AutoDiffControlFlowInterface &
enzyme::getAutoDiffControlFlow(Operation *op) {
  if (Dialect *dialect = op->getDialect())
    if (auto *iface =
            dialect->getRegisteredInterface<AutoDiffControlFlowInterface>())
      return *iface;
  // Default asks no dialect, sets null.
  static const AutoDiffControlFlowInterface defaults(/*dialect=*/nullptr);
  return defaults;
}
