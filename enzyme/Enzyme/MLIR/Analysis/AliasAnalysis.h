#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include <optional>

namespace mlir {

class CallableOpInterface;

namespace enzyme {

//===----------------------------------------------------------------------===//
// AliasClassLattice
//===----------------------------------------------------------------------===//

class AliasClassLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  void print(raw_ostream &os) const override;

  AliasResult alias(const AbstractSparseLattice &other) const;

  std::optional<Value> getCanonicalAllocation() const;
  void getCanonicalAllocations(SmallVectorImpl<Value> &allocations) const;

  ChangeResult join(const AbstractSparseLattice &other) override;

  ChangeResult markFresh();

  ChangeResult markUnknown();

  ChangeResult markEntry();

  ChangeResult reset();

  DenseSet<DistinctAttr> aliasClasses;

  /// Special setting for entry arguments without aliasing information. These
  /// may alias other entry arguments, but will not alias allocations made
  /// within the region.
  bool isEntry = false;

  /// We don't know anything about the aliasing of this value. TODO: re-evaluate
  /// if we need this.
  bool isUnknown = false;

private:
  /// As we compute alias classes, additionally propagate the possible canonical
  /// allocation sites for this
  DenseSet<Value> canonicalAllocations;
};

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

/// This analysis implements interprocedural alias analysis
class AliasAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<AliasClassLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(AliasClassLattice *lattice) override;

  void visitOperation(Operation *op,
                      ArrayRef<const AliasClassLattice *> operands,
                      ArrayRef<AliasClassLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const AliasClassLattice *> operands,
                         ArrayRef<AliasClassLattice *> results) override;
};

} // namespace enzyme
} // namespace mlir

#endif
