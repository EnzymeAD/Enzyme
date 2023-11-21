#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_ALIASANALYSIS_H

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
class AliasClassSet {
public:
  enum class State {
    Undefined, ///< Has not been analyzed yet (lattice bottom).
    Defined,   ///< Has specific alias classes.
    Unknown    ///< Analyzed and may point to any class (lattice top).
  };

  AliasClassSet() : state(State::Undefined) {}

  AliasClassSet(DistinctAttr single) : state(State::Defined) {
    aliasClasses.insert(single);
  }

  // TODO(zinenko): deprecate this and use a visitor instead.
  DenseSet<DistinctAttr> &getAliasClasses() {
    assert(state == State::Defined);
    return aliasClasses;
  }
  const DenseSet<DistinctAttr> &getAliasClasses() const {
    return const_cast<AliasClassSet *>(this)->getAliasClasses();
  }

  bool isUnknown() const { return state == State::Unknown; }
  bool isUndefined() const { return state == State::Undefined; }

  ChangeResult join(const AliasClassSet &other);
  ChangeResult insert(const DenseSet<DistinctAttr> &classes);
  ChangeResult markUnknown();

  static AliasClassSet getFresh(Attribute debugLabel);

  /// Returns true if this set is in the canonical form, i.e. either the state
  /// is `State::Defined` or the explicit list of classes is empty, but not
  /// both.
  bool isCanonical() const;

  /// Returns an instance of AliasClassSet known not to alias with anything.
  /// This is different from "undefined" and "unknown". The instance is *not* a
  /// classical singleton.
  static const AliasClassSet &getEmpty() {
    static const AliasClassSet empty(State::Defined);
    return empty;
  }

  /// Returns an instance of AliasClassSet in "undefined" state, i.e. without a
  /// set of alias classes. This is different from empty alias set, which
  /// indicates that the value is known not to alias with any alias class. The
  /// instance is *not* a classical singleton, there are other ways of obtaining
  /// it.
  static const AliasClassSet &getUndefined() { return undefinedSet; }

  /// Returns an instance of AliasClassSet for the "unknown" class. The instance
  /// is *not* a classical singleton, there are other ways of obtaining an
  /// "unknown" alias set.
  static const AliasClassSet &getUnknown() { return unknownSet; }

  bool operator==(const AliasClassSet &other) const;

  void print(llvm::raw_ostream &os) const;

  ChangeResult
  foreachClass(function_ref<ChangeResult(DistinctAttr, State)> callback) const;

private:
  explicit AliasClassSet(State state) : state(state) {}

  ChangeResult updateStateToDefined() {
    assert(state != State::Unknown && "cannot go back from unknown state");
    ChangeResult result = state == State::Undefined ? ChangeResult::Change
                                                    : ChangeResult::NoChange;
    state = State::Defined;
    return result;
  }

  const static AliasClassSet unknownSet;
  const static AliasClassSet undefinedSet;

  DenseSet<DistinctAttr> aliasClasses;
  State state;
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

  ChangeResult setPointingToEmpty(const AliasClassSet &destClasses);

  /// Mark `dest` as pointing to "unknown" alias set, that is, any possible
  /// other pointer. This is partial pessimistic fixpoint.
  ChangeResult markPointToUnknown(const AliasClassSet &destClasses);

  /// Mark the entire data structure as "unknown", that is, any pointer may be
  /// containing any other pointer. This is the full pessimistic fixpoint.
  ChangeResult markAllPointToUnknown();

  /// Mark all alias classes except the given ones to point to the "unknown"
  /// alias set.
  ChangeResult markAllExceptPointToUnknown(const AliasClassSet &destClasses);

  const AliasClassSet &getPointsTo(DistinctAttr id) const {
    auto it = pointsTo.find(id);
    if (it == pointsTo.end())
      return AliasClassSet::getUndefined();
    return it->getSecond();
  }

private:
  ChangeResult update(const AliasClassSet &keysToUpdate,
                      const AliasClassSet &values, bool replace);

  ChangeResult joinPotentiallyMissing(DistinctAttr key,
                                      const AliasClassSet &value);

  /// Indicates that alias classes not listed as keys in `pointsTo` point to
  /// unknown alias set (when true) or an empty alias set (when false).
  // TODO: consider also differentiating between pointing to known-empty vs.
  // not-yet-computed.
  // bool otherPointToUnknown = false;

  // missing from map always beings "undefined", "unknown"s are stored
  // explicitly.

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

  static AliasClassLattice getFresh(Value point,
                                    Attribute debugLabel = nullptr);

  // ChangeResult markFresh(/*optional=*/Attribute debugLabel);

  ChangeResult markUnknown() { return aliasClasses.markUnknown(); }

  // ChangeResult reset() { return aliasClasses.reset(); }

  /// We don't know anything about the aliasing of this value.
  bool isUnknown() const { return aliasClasses.isUnknown(); }

  bool isUndefined() const { return aliasClasses.isUndefined(); }

  const DenseSet<DistinctAttr> &getAliasClasses() const {
    return aliasClasses.getAliasClasses();
  }

  const AliasClassSet &getAliasClassesObject() const { return aliasClasses; }

private:
  struct UndefinedState {};

  AliasClassLattice(Value value, AliasClassSet &&classes)
      : dataflow::AbstractSparseLattice(value),
        aliasClasses(std::move(classes)) {}

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
