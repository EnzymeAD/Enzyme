//===- DataflowAliasAnalysis.h - Declaration of Alias Analysis ------------===//
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

#include "DataFlowLattice.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

class CallableOpInterface;

namespace enzyme {

/// A set of alias class identifiers to be treated as a single union. May be
/// marked as "unknown", which is a conservative pessimistic state, or as
/// "undefined", which is a "not-yet-analyzed" initial state. Undefined state is
/// different from an empty alias set.
using AliasClassSet = SetLattice<DistinctAttr>;

//===----------------------------------------------------------------------===//
// OriginalClasses
//===----------------------------------------------------------------------===//

/// Alias classes for freshly created, e.g., allocated values. These must
/// be used instead of allocating a fresh distinct attribute every time.
/// Allocation may only happen when the mapping is not already present here.
class OriginalClasses {
public:
  DistinctAttr getOriginalClass(Value value, StringRef debugLabel) {
    return getOriginalClass(value,
                            StringAttr::get(value.getContext(), debugLabel));
  }
  DistinctAttr getOriginalClass(Value value, Attribute referenced = nullptr) {
    DistinctAttr &aliasClass = originalClasses[value];
    if (!aliasClass) {
      if (!referenced)
        referenced = UnitAttr::get(value.getContext());
      aliasClass = DistinctAttr::create(referenced);
    }
    return aliasClass;
  }

  DistinctAttr getSameOriginalClass(ValueRange values, StringRef debugLabel) {
    if (values.empty())
      return nullptr;

    auto label = StringAttr::get(values.front().getContext(), debugLabel);

    DistinctAttr common = nullptr;
    for (Value v : values) {
      DistinctAttr &aliasClass = originalClasses[v];
      if (!aliasClass) {
        if (!common)
          common = DistinctAttr::create(label);
        aliasClass = common;
      } else {
        if (!common)
          common = aliasClass;
        else
          assert(aliasClass == common && "original alias class mismatch");
      }
    }
    return common;
  }

private:
  DenseMap<Value, DistinctAttr> originalClasses;
};

//===----------------------------------------------------------------------===//
// PointsToSets
//
// Specifically for pointers to pointers. This tracks alias information through
// pointers stored/loaded through memory.
//===----------------------------------------------------------------------===//

class PointsToSets : public MapOfSetsLattice<DistinctAttr, DistinctAttr> {
public:
  using MapOfSetsLattice::MapOfSetsLattice;

  void print(raw_ostream &os) const override;

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

  ChangeResult setPointingToEmpty(const AliasClassSet &destClasses);

  /// Mark `dest` as pointing to "unknown" alias set, that is, any possible
  /// other pointer. This is partial pessimistic fixpoint.
  ChangeResult markPointToUnknown(const AliasClassSet &destClasses);

  /// Mark the entire data structure as "unknown", that is, any pointer may be
  /// containing any other pointer. This is the full pessimistic fixpoint.
  ChangeResult markAllPointToUnknown() { return markAllUnknown(); }

  /// Mark all alias classes except the given ones to point to the "unknown"
  /// alias set.
  ChangeResult markAllExceptPointToUnknown(const AliasClassSet &destClasses);

  const AliasClassSet &getPointsTo(DistinctAttr id) const { return lookup(id); }

  //void dumpSets() const;
private:
  /// Update all alias classes in `keysToUpdate` to additionally point to alias
  /// classes in `values`. Handle undefined keys optimistically (ignore) and
  /// unknown keys pessimistically (update all existing keys). `replace` is a
  /// debugging aid that indicates whether the update is intended to replace the
  /// pre-existing state, it has no effect in NoAsserts build. Since we don't
  /// want to forcefully reset pointsTo value as that is not guaranteed to make
  /// monotonous progress on the lattice and therefore convergence to fixpoint,
  /// replacement is only expected for a previously "unknown" value (absent from
  /// the mapping) or for a value with itself. Replacement is therefore handled
  /// as a regular update, i.e. join, with additional assertions. Note that
  /// currently an update is possible to _any_ value that is >= the current one
  /// in the lattice, not only the replacements described above.
  ChangeResult update(const AliasClassSet &keysToUpdate,
                      const AliasClassSet &values, bool replace);

};

//===----------------------------------------------------------------------===//
// PointsToPointerAnalysis
//===----------------------------------------------------------------------===//

class PointsToPointerAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<PointsToSets> {
public:
  PointsToPointerAnalysis(DataFlowSolver &solver)
      : DenseForwardDataFlowAnalysis(solver) {}

  void setToEntryState(PointsToSets *lattice) override;

  LogicalResult visitOperation(Operation *op, const PointsToSets &before,
                               PointsToSets *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const PointsToSets &before,
                                    PointsToSets *after) override;

  void processCapturingStore(ProgramPoint *dependent, PointsToSets *after,
                             Value capturedValue, Value destinationAddress,
                             bool isMustStore = false);

  void processCallToSummarizedFunc(
      CallOpInterface call,
      const DenseMap<DistinctAttr, AliasClassSet> &summary,
      PointsToSets *after);

private:
  /// Alias classes originally assigned to known-distinct values, e.g., fresh
  /// allocations, by this analysis. This does NOT necessarily need to be shared
  /// with the other analysis as they may assign different classes, e.g., for
  /// results of the same call.
  OriginalClasses originalClasses;
};

//===----------------------------------------------------------------------===//
// AliasClassLattice
//===----------------------------------------------------------------------===//

class AliasClassLattice : public SparseSetLattice<DistinctAttr> {
public:
  using SparseSetLattice::SparseSetLattice;

  void print(raw_ostream &os) const override;

  ::mlir::AliasResult alias(const AbstractSparseLattice &other) const;

  ChangeResult join(const AbstractSparseLattice &other) override;

  static AliasClassLattice single(Value point, DistinctAttr value) {
    return AliasClassLattice(point, AliasClassSet(value));
  }

  const DenseSet<DistinctAttr> &getAliasClasses() const {
    return elements.getElements();
  }

  const AliasClassSet &getAliasClassesObject() const { return elements; }
};

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

/// This analysis implements interprocedural alias analysis
class AliasAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<AliasClassLattice> {
public:
  AliasAnalysis(DataFlowSolver &solver, MLIRContext *ctx, bool relative = false)
      : SparseForwardDataFlowAnalysis(solver),
        entryClass(DistinctAttr::create(StringAttr::get(ctx, "entry"))),
        relative(relative) {
    if (relative)
      assert(!solver.getConfig().isInterprocedural());
  }

  void setToEntryState(AliasClassLattice *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const AliasClassLattice *> operands,
                               ArrayRef<AliasClassLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const AliasClassLattice *> operands,
                         ArrayRef<AliasClassLattice *> results) override;

private:
  void transfer(Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects,
                ArrayRef<const AliasClassLattice *> operands,
                ArrayRef<AliasClassLattice *> results);

  /// Create a pseudo alias class when loading from a function argument that
  /// points to unknown locations. The pseudo class indicates that it points to
  /// _something_ and is expected to be unified with a concrete alias class when
  /// the function summaries are used at this function's call sites.
  void createImplicitArgDereference(Operation *op, AliasClassLattice *source,
                                    DistinctAttr srcClass,
                                    AliasClassLattice *result);

  /// A special alias class to denote unannotated pointer arguments.
  const DistinctAttr entryClass;

  /// If true, the analysis will operate in an intraprocedural way assuming it
  /// is called bottom-up on the function call graph.
  const bool relative;

  /// Alias classes originally assigned to known-distinct values, e.g., fresh
  /// allocations, by this analysis. This does NOT necessarily need to be shared
  /// with the other analysis as they may assign different classes, e.g., for
  /// results of the same call.
  OriginalClasses originalClasses;
};

} // namespace enzyme
} // namespace mlir

#endif
