#ifndef ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H
#define ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H

#include "AliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Lattice.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
class FunctionOpInterface;

namespace enzyme {

using ValueOriginSet = SetLattice<Attribute>;

//===----------------------------------------------------------------------===//
// ValueOriginsLattice
//===----------------------------------------------------------------------===//

class ValueOriginsLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;
  ValueOriginsLattice(Value value, ValueOriginSet &&origins)
      : dataflow::AbstractSparseLattice(value), origins(std::move(origins)) {}

  static ValueOriginsLattice single(Value point, Attribute value) {
    return ValueOriginsLattice(point, ValueOriginSet(value));
  }

  void print(raw_ostream &os) const override;

  ChangeResult join(const AbstractSparseLattice &other) override;

  ChangeResult join(const ValueOriginSet &other) { return origins.join(other); }

  ChangeResult insert(const DenseSet<Attribute> &classes) {
    return origins.insert(classes);
  }

  ChangeResult markUnknown() { return origins.markUnknown(); }

  bool isUnknown() const { return origins.isUnknown(); }

  bool isUndefined() const { return origins.isUndefined(); }

  const DenseSet<Attribute> &getOrigins() const {
    return origins.getElements();
  }

  const ValueOriginSet &getOriginsObject() const { return origins; }

private:
  ValueOriginSet origins;
};

// TODO: create a common inherited class?
class BackwardOriginsLattice : public ValueOriginsLattice {
public:
  using ValueOriginsLattice::ValueOriginsLattice;

  ChangeResult meet(const AbstractSparseLattice &other) override {
    // MLIR framework again misusing terminology
    return join(other);
  }
};

class ForwardActivityAnnotationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ValueOriginsLattice> {
public:
  ForwardActivityAnnotationAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver) {
    assert(!solver.getConfig().isInterprocedural());
  }

  void setToEntryState(ValueOriginsLattice *lattice) override;

  void visitOperation(Operation *op,
                      ArrayRef<const ValueOriginsLattice *> operands,
                      ArrayRef<ValueOriginsLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const ValueOriginsLattice *> operands,
                         ArrayRef<ValueOriginsLattice *> results) override;

private:
  void processMemoryRead(Operation *op, Value address,
                         ArrayRef<ValueOriginsLattice *> results);

  OriginalClasses originalClasses;
};

class BackwardActivityAnnotationAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<BackwardOriginsLattice> {
public:
  BackwardActivityAnnotationAnalysis(DataFlowSolver &solver,
                                     SymbolTableCollection &symbolTable)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable) {
    assert(!solver.getConfig().isInterprocedural());
  }

  void visitBranchOperand(OpOperand &operand) override {}

  void visitCallOperand(OpOperand &operand) override {}

  void setToExitState(BackwardOriginsLattice *lattice) override;

  void
  visitOperation(Operation *op, ArrayRef<BackwardOriginsLattice *> operands,
                 ArrayRef<const BackwardOriginsLattice *> results) override;

  void
  visitExternalCall(CallOpInterface call,
                    ArrayRef<BackwardOriginsLattice *> operands,
                    ArrayRef<const BackwardOriginsLattice *> results) override;
};

//===----------------------------------------------------------------------===//
// ValueOriginsMap
//===----------------------------------------------------------------------===//

class ValueOriginsMap : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  void print(raw_ostream &os) const override;

  Attribute serialize(MLIRContext *ctx) const;

  ChangeResult join(const AbstractDenseLattice &other) override;

  /// Mark the pointer stored in `dest` as originating from all of `origins`.
  ChangeResult insert(const AliasClassSet &destClasses,
                      const ValueOriginSet &origins);

  ChangeResult markAllOriginsUnknown();

  ChangeResult joinPotentiallyMissing(DistinctAttr key,
                                      const ValueOriginSet &value);

  const ValueOriginSet &getOrigins(DistinctAttr id) const {
    auto it = valueOrigins.find(id);
    if (it == valueOrigins.end())
      return ValueOriginSet::getUndefined();
    return it->getSecond();
  }

private:
  // Represents "this alias class has a differential dependency originating from
  // this value"
  // TODO: Don't get confused because they're both distinct attributes, the keys
  // are exclusively alias classes and the values are sets of value origins
  DenseMap<DistinctAttr, ValueOriginSet> valueOrigins;
};

// TODO: reconcile the origins vs origin in the naming
class BackwardValueOriginMap : public ValueOriginsMap {
public:
  using ValueOriginsMap::ValueOriginsMap;

  ChangeResult meet(const AbstractDenseLattice &other) override {
    return join(other);
  }
};

//===----------------------------------------------------------------------===//
// DenseActivityAnnotationAnalysis
//===----------------------------------------------------------------------===//

class DenseActivityAnnotationAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<ValueOriginsMap> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void setToEntryState(ValueOriginsMap *lattice) override;

  void visitOperation(Operation *op, const ValueOriginsMap &before,
                      ValueOriginsMap *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const ValueOriginsMap &before,
                                    ValueOriginsMap *after) override;

private:
  void processCallToSummarizedFunc(
      CallOpInterface call,
      const DenseMap<DistinctAttr, ValueOriginSet> &summary,
      const ValueOriginsMap &before, ValueOriginsMap *after);

  void processCopy(Operation *op, Value copySource, Value copyDest,
                   const ValueOriginsMap &before, ValueOriginsMap *after);

  OriginalClasses originalClasses;
};

class DenseBackwardActivityAnnotationAnalysis
    : public dataflow::DenseBackwardDataFlowAnalysis<BackwardValueOriginMap> {
public:
  using DenseBackwardDataFlowAnalysis::DenseBackwardDataFlowAnalysis;

  void visitOperation(Operation *op, const BackwardValueOriginMap &after,
                      BackwardValueOriginMap *before) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const BackwardValueOriginMap &after,
                                    BackwardValueOriginMap *before) override;

  void setToExitState(BackwardValueOriginMap *lattice) override;
};

void runActivityAnnotations(FunctionOpInterface callee);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H
