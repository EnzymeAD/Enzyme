//===- DataFlowActivityAnalysis.h - Declaration of Activity Analysis ------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file formulates activity analysis within
// a dataflow framework.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_ACTIVITYANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_ACTIVITYANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "DataFlowAliasAnalysis.h"
#include "Dialect/Ops.h"

#include "Interfaces/AutoDiffOpInterface.h"

using namespace mlir::dataflow;
namespace mlir {
class FunctionOpInterface;

namespace enzyme {

enum class Activity : uint32_t;

/// From LLVM Enzyme's activity analysis, there are four activity states.
// constant instruction vs constant value, a value/instruction (one and the same
// in LLVM) can be a constant instruction but active value, active instruction
// but constant value, or active/constant both.

// The result of activity states are potentially different for multiple
// enzyme.autodiff calls.
enum class ActivityKind { Constant, ActiveVal, Unknown };

class ValueActivity {
public:
  static ValueActivity getConstant() {
    return ValueActivity(ActivityKind::Constant);
  }

  static ValueActivity getActiveVal() {
    return ValueActivity(ActivityKind::ActiveVal);
  }

  static ValueActivity getUnknown() {
    return ValueActivity(ActivityKind::Unknown);
  }

  bool isActiveVal() const { return value == ActivityKind::ActiveVal; }

  bool isConstant() const { return value == ActivityKind::Constant; }

  bool isUnknown() const { return value == ActivityKind::Unknown; }

  ValueActivity() {}
  ValueActivity(ActivityKind value) : value(value) {}

  /// Get the known activity state.
  const ActivityKind &getValue() const { return value; }

  bool operator==(const ValueActivity &rhs) const { return value == rhs.value; }

  static ValueActivity merge(const ValueActivity &lhs,
                             const ValueActivity &rhs) {
    if (lhs.isUnknown() || rhs.isUnknown())
      return ValueActivity::getUnknown();

    if (lhs.isConstant() && rhs.isConstant())
      return ValueActivity::getConstant();
    return ValueActivity::getActiveVal();
  }

  static ValueActivity join(const ValueActivity &lhs,
                            const ValueActivity &rhs) {
    return ValueActivity::merge(lhs, rhs);
  }

  void print(raw_ostream &os) const;
  raw_ostream &operator<<(raw_ostream &os) const;

private:
  /// The activity kind. Optimistically initialized to constant.
  ActivityKind value = ActivityKind::Constant;
};

//===----------------------------------------------------------------------===//
// ForwardValueActivity
//===----------------------------------------------------------------------===//
class ForwardValueActivity : public Lattice<enzyme::ValueActivity> {
public:
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// BackwardValueActivity
//===----------------------------------------------------------------------===//
class BackwardValueActivity : public AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  enzyme::ValueActivity getValue() const { return value; }

  void print(raw_ostream &os) const override;

  ChangeResult meet(const AbstractSparseLattice &other) override;

  ChangeResult meet(enzyme::ValueActivity other);

private:
  enzyme::ValueActivity value;
};

/// Sparse activity analysis reasons about activity by traversing forward down
/// the def-use chains starting from active function arguments.
class SparseForwardActivityAnalysis
    : public SparseForwardDataFlowAnalysis<enzyme::ForwardValueActivity> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(enzyme::ForwardValueActivity *lattice) override;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const enzyme::ForwardValueActivity *> operands,
                 ArrayRef<enzyme::ForwardValueActivity *> results) override;

  void
  visitExternalCall(CallOpInterface call,
                    ArrayRef<const enzyme::ForwardValueActivity *> operands,
                    ArrayRef<enzyme::ForwardValueActivity *> results) override;

  void transfer(Operation *op,
                ArrayRef<const enzyme::ForwardValueActivity *> operands,
                ArrayRef<enzyme::ForwardValueActivity *> results);
};

class SparseBackwardActivityAnalysis
    : public SparseBackwardDataFlowAnalysis<enzyme::BackwardValueActivity> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void setToExitState(enzyme::BackwardValueActivity *lattice) override {
    // llvm::errs() << "backward sparse setting to exit state\n";
  }

  void visitBranchOperand(OpOperand &operand) override {}

  void visitCallOperand(OpOperand &operand) override {}

  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}

  void transfer(Operation *op,
                ArrayRef<enzyme::BackwardValueActivity *> operands,
                ArrayRef<const enzyme::BackwardValueActivity *> results);

  LogicalResult visitOperation(
      Operation *op, ArrayRef<enzyme::BackwardValueActivity *> operands,
      ArrayRef<const enzyme::BackwardValueActivity *> results) override;

  void visitExternalCall(
      CallOpInterface call, ArrayRef<enzyme::BackwardValueActivity *> operands,
      ArrayRef<const enzyme::BackwardValueActivity *> results) override;
};

/// This needs to keep track of three things:
///   1. Could active info store in?
///   2. Could active info load out?
///   TODO: Necessary for run-time activity
///   3. Could constant info propagate (store?) in?
///
/// Active: (forward) active in && (backward) active out && (??) !const in
/// ActiveOrConstant: active in && active out && const in
/// Constant: everything else
struct MemoryActivityState {
  /// Whether active data has stored into this memory location.
  bool activeIn = false;
  /// Whether active data was loaded out of this memory location.
  bool activeOut = false;

  bool operator==(const MemoryActivityState &other) {
    return activeIn == other.activeIn && activeOut == other.activeOut;
  }

  bool operator!=(const MemoryActivityState &other) {
    return !(*this == other);
  }

  ChangeResult reset();
  ChangeResult merge(const MemoryActivityState &other);
};

class MemoryActivity : public AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  /// Clear all modifications.
  ChangeResult reset();

  bool hasActiveData(DistinctAttr aliasClass) const;

  bool activeDataFlowsOut(DistinctAttr aliasClass) const;

  /// Set the internal activity state. Accepts null attribute to indicate "other
  /// classes".
  ChangeResult setActiveIn(DistinctAttr aliasClass);
  ChangeResult setActiveIn();
  ChangeResult setActiveOut(DistinctAttr aliasClass);
  ChangeResult setActiveOut();
  void print(raw_ostream &os) const override;
  raw_ostream &operator<<(raw_ostream &os) const;

protected:
  ChangeResult merge(const AbstractDenseLattice &lattice);

private:
  DenseMap<DistinctAttr, MemoryActivityState> activityStates;
  MemoryActivityState otherMemoryActivity;
};

class ForwardMemoryActivity : public MemoryActivity {
public:
  using MemoryActivity::MemoryActivity;

  /// Join the activity states.
  ChangeResult join(const AbstractDenseLattice &lattice) {
    return merge(lattice);
  }
};

class BackwardMemoryActivity : public MemoryActivity {
public:
  using MemoryActivity::MemoryActivity;

  ChangeResult meet(const AbstractDenseLattice &lattice) override {
    return merge(lattice);
  }
};

class DenseForwardActivityAnalysis
    : public DenseForwardDataFlowAnalysis<ForwardMemoryActivity> {
public:
  DenseForwardActivityAnalysis(DataFlowSolver &solver, Block *entryBlock,
                               ArrayRef<enzyme::Activity> argumentActivity)
      : DenseForwardDataFlowAnalysis(solver), entryBlock(entryBlock),
        argumentActivity(argumentActivity) {}

  LogicalResult visitOperation(Operation *op,
                               const ForwardMemoryActivity &before,
                               ForwardMemoryActivity *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const ForwardMemoryActivity &before,
                                    ForwardMemoryActivity *after) override {
    join(after, before);
  }

  /// Initialize the entry block with the supplied argument activities.
  void setToEntryState(ForwardMemoryActivity *lattice) override;

private:
  // A pointer to the entry block and argument activities of the top-level
  // function being differentiated. This is used to set the entry state
  // because we need access to the results of points-to analysis.
  Block *entryBlock;
  SmallVector<enzyme::Activity> argumentActivity;
};

class DenseBackwardActivityAnalysis
    : public DenseBackwardDataFlowAnalysis<BackwardMemoryActivity> {
public:
  DenseBackwardActivityAnalysis(DataFlowSolver &solver,
                                SymbolTableCollection &symbolTable,
                                FunctionOpInterface parentOp,
                                ArrayRef<enzyme::Activity> argumentActivity,
                                ArrayRef<enzyme::Activity> returnActivity)
      : DenseBackwardDataFlowAnalysis(solver, symbolTable), parentOp(parentOp),
        argumentActivity(argumentActivity), returnActivity(returnActivity) {}

  LogicalResult visitOperation(Operation *op,
                               const BackwardMemoryActivity &after,
                               BackwardMemoryActivity *before) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const BackwardMemoryActivity &after,
                                    BackwardMemoryActivity *before) override {
    meet(before, after);
  }

  void setToExitState(BackwardMemoryActivity *lattice) override {}

private:
  FunctionOpInterface parentOp;
  SmallVector<enzyme::Activity> argumentActivity;
  SmallVector<enzyme::Activity> returnActivity;
};

void runDataFlowActivityAnalysis(FunctionOpInterface callee,
                                 ArrayRef<enzyme::Activity> argumentActivity,
                                 ArrayRef<enzyme::Activity> returnActivity,
                                 bool print = false, bool verbose = false,
                                 bool annotate = false);

} // namespace enzyme
} // namespace mlir

#endif
