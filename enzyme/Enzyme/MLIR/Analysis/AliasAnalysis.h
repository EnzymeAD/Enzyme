//===- AliasAnalysis.h - Declaration of Alias Analysis --------------------===//
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
// This file contains the declaration of Alias (and Points-To) Analysis, a
// general analysis that determines the possible static memory locations
// that the pointers in a program may point to.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

class CallableOpInterface;

namespace enzyme {

/// A set of alias class identifiers to be treated as a single union. May be
/// marked as "unknown", which is a conservative pessimistic state.
struct AliasClassSet {
  AliasClassSet() = default;
  AliasClassSet(DistinctAttr single) { aliasClasses.insert(single); }

  DenseSet<DistinctAttr> &getAliasClasses() {
    assert(!unknown);
    return aliasClasses;
  }
  const DenseSet<DistinctAttr> &getAliasClasses() const {
    return const_cast<AliasClassSet *>(this)->getAliasClasses();
  }

  bool isUnknown() const { return unknown; }

  ChangeResult join(const AliasClassSet &other);
  ChangeResult insert(const DenseSet<DistinctAttr> &classes);
  ChangeResult markUnknown();
  ChangeResult markFresh(Attribute debugLabel);
  ChangeResult reset();

  /// Returns true if this set is in the canonical form, i.e. has either the
  /// unknown bit or the explicit list of classes, but not both.
  bool isCanonical() const;

  /// Returns an instance of AliasClassSet for the "unknown" class. The instance
  /// is *not* a classical singleton, there are other ways of obtaining an
  /// "unknown" alias set.
  static const AliasClassSet &getUnknown() { return unknownSet; }

  bool operator==(const AliasClassSet &other) const;

  ChangeResult
  foreachClass(function_ref<ChangeResult(DistinctAttr)> callback) const;

private:
  explicit AliasClassSet(bool unknown) : unknown(unknown) {}

  const static AliasClassSet unknownSet;

  DenseSet<DistinctAttr> aliasClasses;
  bool unknown = false;
};

//===----------------------------------------------------------------------===//
// PointsToAnalysis
//
// Specifically for pointers to pointers. This tracks alias information through
// pointers stored/loaded through memory.
//===----------------------------------------------------------------------===//

class PointsToSets : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  void print(raw_ostream &os) const override;

  ChangeResult join(const AbstractDenseLattice &lattice) override;

  /// Mark the pointer stored in `dest` as possibly pointing to any of `values`,
  /// instead of the values it may be currently pointing to.
  ChangeResult setPointingToClasses(const AliasClassSet &destClasses,
                                    const AliasClassSet &values) {
    return update(destClasses, values, /*replace=*/true);
  }

  /// Mark the pointer stored in `dest` as possibly pointing to any of `values`,
  /// in addition to the values it may already point to.
  ChangeResult insert(const AliasClassSet &destClasses,
                      const AliasClassSet &values) {
    return update(destClasses, values, /*replace=*/false);
  };

  /// For every alias class in `dest`, record that it may additionally be
  /// pointing to the same as the classes in `src`.
  ChangeResult addSetsFrom(const AliasClassSet &destClasses,
                           const AliasClassSet &srcClasses);

  /// For every alias class in `dest`, record that it is pointing to the _same_
  /// new alias set.
  ChangeResult setPointingToFresh(const AliasClassSet &destClasses,
                                  StringAttr debugLabel);

  /// Mark `dest` as pointing to "unknown" alias set, that is, any possible
  /// other pointer. This is partial pessimistic fixpoint.
  ChangeResult markPointToUnknown(const AliasClassSet &destClasses);

  /// Mark the entire data structure as "unknown", that is, any pointer may be
  /// containing any other pointer. This is the full pessimistic fixpoint.
  ChangeResult markAllPointToUnknown();

  const AliasClassSet &getPointsTo(DistinctAttr id) const {
    auto it = pointsTo.find(id);
    if (it == pointsTo.end())
      return AliasClassSet::getUnknown();
    return it->getSecond();
  }

private:
  ChangeResult update(const AliasClassSet &keysToUpdate,
                      const AliasClassSet &values, bool replace);

  /// Indicates that alias classes not listed as keys in `pointsTo` point to
  /// unknown alias set (when true) or an empty alias set (when false).
  // TODO: consider also differentiating between pointing to known-empty vs.
  // not-yet-computed.
  bool otherPointToUnknown = false;

  /// Maps an identifier of an alias set to the set of alias sets its value may
  /// belong to. When an identifier is not present in this map, it is considered
  /// to point to either the unknown set or nothing, based on the value of
  /// `otherPointToUnknown`.
  DenseMap<DistinctAttr, AliasClassSet> pointsTo;
};

class PointsToPointerAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<PointsToSets> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void setToEntryState(PointsToSets *lattice) override;

  void visitOperation(Operation *op, const PointsToSets &before,
                      PointsToSets *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const PointsToSets &before,
                                    PointsToSets *after) override;

  void processCapturingStore(ProgramPoint dependent, PointsToSets *after,
                             Value capturedValue, Value destinationAddress,
                             bool isMustStore = false);
};

//===----------------------------------------------------------------------===//
// AliasClassLattice
//===----------------------------------------------------------------------===//

class AliasClassLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  void print(raw_ostream &os) const override;

  AliasResult alias(const AbstractSparseLattice &other) const;

  ChangeResult join(const AbstractSparseLattice &other) override;

  ChangeResult insert(const DenseSet<DistinctAttr> &classes) {
    return aliasClasses.insert(classes);
  }

  ChangeResult markFresh(/*optional=*/Attribute debugLabel);

  ChangeResult markUnknown() { return aliasClasses.markUnknown(); }

  ChangeResult reset() { return aliasClasses.reset(); }

  static DistinctAttr getFresh(Attribute debugLabel) {
    return DistinctAttr::create(debugLabel);
  }

  /// We don't know anything about the aliasing of this value. TODO: re-evaluate
  /// if we need this.
  bool isUnknown() const { return aliasClasses.isUnknown(); }

  const DenseSet<DistinctAttr> &getAliasClasses() const {
    return aliasClasses.getAliasClasses();
  }

  const AliasClassSet &getAliasClassesObject() const { return aliasClasses; }

private:
  AliasClassSet aliasClasses;
};

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

/// This analysis implements interprocedural alias analysis
class AliasAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<AliasClassLattice> {
public:
  AliasAnalysis(DataFlowSolver &solver, MLIRContext *ctx)
      : SparseForwardDataFlowAnalysis(solver),
        entryClass(DistinctAttr::create(StringAttr::get(ctx, "entry"))) {}

  void setToEntryState(AliasClassLattice *lattice) override;

  void visitOperation(Operation *op,
                      ArrayRef<const AliasClassLattice *> operands,
                      ArrayRef<AliasClassLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const AliasClassLattice *> operands,
                         ArrayRef<AliasClassLattice *> results) override;

private:
  void transfer(Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects,
                ArrayRef<const AliasClassLattice *> operands,
                ArrayRef<AliasClassLattice *> results);

  /// A special alias class to denote unannotated pointer arguments.
  const DistinctAttr entryClass;
};

} // namespace enzyme
} // namespace mlir

#endif
