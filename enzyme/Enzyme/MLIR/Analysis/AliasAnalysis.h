#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <optional>

namespace mlir {

class CallableOpInterface;

namespace enzyme {

/// A set of alias class identifiers to be treated as a single union. May be
/// marked as "unknown", which is a conservative pessimistic state.
struct AliasClassSet {
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

private:
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

  ChangeResult insert(DistinctAttr dest, const AliasClassSet &values);
  ChangeResult insertFresh(DistinctAttr dest, StringAttr debugLabel = nullptr);

  /// Mark `dest` as pointing to "unknown" alias set, that is, any possible
  /// other pointer. This is partial pessimistic fixpoint.
  ChangeResult markUnknown(DistinctAttr dest);

  /// Mark the entire data structure as "unknown", that is, any pointer may be
  /// containing any other pointer. This is the full pessimistic fixpoint.
  ChangeResult markUnknown();

  DenseMap<DistinctAttr, AliasClassSet> pointsTo;

private:
  bool unknown = false;
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
                             Value capturedValue, Value destinationAddress);
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
