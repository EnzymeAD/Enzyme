//===- AutoDiffControlFlowInterface.h - Control flow for reverse mode -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dialect interface through which reverse mode builds
// the control flow of its reverse sweep: the branches between reverse blocks,
// and the index each block records for the predecessor it was entered from.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_AUTODIFFCONTROLFLOWINTERFACE_H
#define ENZYME_MLIR_INTERFACES_AUTODIFFCONTROLFLOWINTERFACE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectInterface.h"

namespace mlir {
namespace enzyme {

// How a dialect spells the reverse sweep's control flow. A dialect whose
// functions cannot hold cf and arith ops registers its own; the defaults here
// use cf and arith.
class AutoDiffControlFlowInterface
    : public DialectInterface::Base<AutoDiffControlFlowInterface> {
public:
  // Public so that the defaults can be built without a dialect.
  AutoDiffControlFlowInterface(Dialect *dialect) : Base(dialect) {}

  virtual Type getBranchIndexType(MLIRContext *context) const;

  virtual Value createBranchIndex(OpBuilder &builder, Location loc,
                                  int64_t value) const;

  virtual Operation *createBranch(OpBuilder &builder, Location loc,
                                  Block *dest) const;

  // A branch to caseDests[i] when flag equals caseValues[i], and to
  // defaultDest otherwise.
  virtual Operation *createSwitch(OpBuilder &builder, Location loc, Value flag,
                                  Block *defaultDest,
                                  ArrayRef<int64_t> caseValues,
                                  BlockRange caseDests) const;

  // if True when the branch index flag equals value, and ifFalse otherwise.
  virtual Value createSelectOnBranchIndex(OpBuilder &builder, Location loc,
                                          Value flag, int64_t value,
                                          Value ifTrue, Value ifFalse) const;
};

// The interface registered by the dialect of op, or the cf and arith defaults
// when it registers none.
const AutoDiffControlFlowInterface &getAutoDiffControlFlow(Operation *op);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_AUTODIFFCONTROLFLOWINTERFACE_H
