#include "DataFlowActivityAnalysis.h"
#include "AliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// TODO: Don't depend on specific dialects
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"

using namespace mlir;
using namespace mlir::dataflow;
using enzyme::AliasClassLattice;

/// From LLVM Enzyme's activity analysis, there are four activity states.
// constant instruction vs constant value, a value/instruction (one and the same
// in LLVM) can be a constant instruction but active value, active instruction
// but constant value, or active/constant both.

// The result of activity states are potentially different for multiple
// enzyme.autodiff calls.

enum class ActivityKind { ActiveVal, ActivePtr, Constant, Unknown };

using llvm::errs;
class ValueActivity {
public:
  static ValueActivity getConstant() {
    return ValueActivity(ActivityKind::Constant);
  }

  static ValueActivity getActiveVal() {
    return ValueActivity(ActivityKind::ActiveVal);
  }

  static ValueActivity getActivePtr() {
    return ValueActivity(ActivityKind::ActivePtr);
  }

  static ValueActivity getUnknown() {
    return ValueActivity(ActivityKind::Unknown);
  }

  bool isActiveVal() const {
    return value.has_value() && *value == ActivityKind::ActiveVal;
  }

  bool isActivePtr() const {
    return value.has_value() && *value == ActivityKind::ActivePtr;
  }

  bool isConstant() const {
    return value.has_value() && *value == ActivityKind::Constant;
  }

  bool isUnknown() const {
    return value.has_value() && *value == ActivityKind::Unknown;
  }

  ValueActivity(std::optional<ActivityKind> value = std::nullopt)
      : value(std::move(value)) {}

  /// Whether the activity state is uninitialized. This happens when the state
  /// hasn't been set during the analysis.
  bool isUninitialized() const { return !value.has_value(); }

  /// Get the known activity state.
  const ActivityKind &getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  bool operator==(const ValueActivity &rhs) const { return value == rhs.value; }

  static ValueActivity merge(const ValueActivity &lhs,
                             const ValueActivity &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs.isUnknown() || rhs.isUnknown())
      return ValueActivity::getUnknown();

    // We can't merge an active value with an active pointer
    if ((lhs.isActivePtr() && rhs.isActiveVal()) ||
        (rhs.isActiveVal() && lhs.isActivePtr()))
      return ValueActivity::getUnknown();

    if (lhs.isConstant() && rhs.isConstant())
      return ValueActivity::getConstant();

    // Active Val + Constant = Active Val
    // Active Ptr + Constant = Active Ptr
    if (lhs.isActiveVal() || rhs.isActiveVal()) {
      return ValueActivity::getActiveVal();
    }
    return ValueActivity::getActivePtr();
  }

  static ValueActivity join(const ValueActivity &lhs,
                            const ValueActivity &rhs) {
    return ValueActivity::merge(lhs, rhs);
  }

  void print(raw_ostream &os) const {
    if (!value) {
      os << "<uninitialized>";
      return;
    }
    switch (*value) {
    case ActivityKind::ActiveVal:
      os << "ActiveVal";
      break;
    case ActivityKind::ActivePtr:
      os << "ActivePtr";
      break;
    case ActivityKind::Constant:
      os << "Constant";
      break;
    case ActivityKind::Unknown:
      os << "Unknown";
      break;
    }
  }

  raw_ostream &operator<<(raw_ostream &os) const {
    print(os);
    return os;
  }

private:
  /// The known activity kind.
  std::optional<ActivityKind> value;
};

raw_ostream &operator<<(raw_ostream &os, const ValueActivity &val) {
  val.print(os);
  return os;
}

class ForwardValueActivity : public Lattice<ValueActivity> {
public:
  using Lattice::Lattice;
};

class BackwardValueActivity : public AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  ChangeResult meet(const AbstractSparseLattice &other) override {
    const auto *rhs = reinterpret_cast<const BackwardValueActivity *>(&other);
    return meet(rhs->getValue());
  }

  void print(raw_ostream &os) const override { value.print(os); }

  ValueActivity getValue() const { return value; }

  ChangeResult meet(ValueActivity other) {
    auto met = ValueActivity::merge(getValue(), other);
    if (getValue() == met) {
      return ChangeResult::NoChange;
    }

    value = met;
    return ChangeResult::Change;
  }

private:
  ValueActivity value;
};

// A trivial dense dataflow analysis to record all allocations available at the
// current program point. This is used if alias analysis doesn't know what the
// canonical allocations of a given pointer are, which occurs with function
// entry arguments and pointers loaded from memory.
class AvailableAllocations : public AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult reset() {
    if (allocations.empty())
      return ChangeResult::NoChange;
    allocations.clear();
    return ChangeResult::Change;
  }

  void getAllocations(SmallVectorImpl<Value> &out) const {
    out.append(allocations.begin(), allocations.end());
  }

  ChangeResult addAllocation(Value value) {
    if (allocations.contains(value))
      return ChangeResult::NoChange;

    allocations.insert(value);
    return ChangeResult::Change;
  }

  ChangeResult join(const AbstractDenseLattice &lattice) override {
    const auto &rhs = static_cast<const AvailableAllocations &>(lattice);
    size_t oldAllocSize = allocations.size();
    allocations.insert(rhs.allocations.begin(), rhs.allocations.end());
    return oldAllocSize == allocations.size() ? ChangeResult::NoChange
                                              : ChangeResult::Change;
  }

  void print(raw_ostream &os) const override {
    os << "{";
    llvm::interleaveComma(allocations, os);
    os << "}\n";
  }

private:
  DenseSet<Value> allocations;
};

raw_ostream &operator<<(raw_ostream &os, const CallControlFlowAction &action) {
  switch (action) {
  case CallControlFlowAction::EnterCallee:
    os << "EnterCallee";
    break;
  case CallControlFlowAction::ExitCallee:
    os << "ExitCallee";
    break;
  case CallControlFlowAction::ExternalCallee:
    os << "ExternalCallee";
    break;
  }
  return os;
}

class AvailableAllocAnalysis
    : public DenseForwardDataFlowAnalysis<AvailableAllocations> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, const AvailableAllocations &before,
                      AvailableAllocations *after) override {
    join(after, before);

    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memory) {
      return;
    }

    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);
    for (const auto &effect : effects) {
      Value value = effect.getValue();

      if (!value)
        return;

      if (isa<MemoryEffects::Allocate>(effect.getEffect()))
        propagateIfChanged(after, after->addAllocation(value));
    }
  }

  void setToEntryState(AvailableAllocations *lattice) override {}

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const AvailableAllocations &before,
                                    AvailableAllocations *after) override {
    if (action == CallControlFlowAction::ExternalCallee) {
      // Attempt to infer the language semantics from the name
      // TODO: Add these semantics in a shared spot with alias analysis
      auto symbol = cast<SymbolRefAttr>(call.getCallableForCallee());
      if (symbol.getLeafReference().getValue() == "malloc") {
        assert(call->getNumResults() == 1);
        propagateIfChanged(after, after->addAllocation(call->getResult(0)));
      }
    }
    join(after, before);
  }
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
  bool activeLoad;
  bool activeStore;
  // Active init is like active store, but a special case for arguments. We
  // need to distinguish arguments that start with active data vs arguments
  // that get active data stored into them during the function.
  bool activeInit;
  // Analogous special case for arguments that are written to, thus having
  // active data escape
  bool activeEscape;

  bool operator==(const MemoryActivityState &other) {
    return activeLoad == other.activeLoad && activeStore == other.activeStore &&
           activeInit == other.activeInit && activeEscape == other.activeEscape;
  }

  bool operator!=(const MemoryActivityState &other) {
    return !(*this == other);
  }

  ChangeResult merge(const MemoryActivityState &other) {
    if (*this == other) {
      return ChangeResult::NoChange;
    }

    activeLoad |= other.activeLoad;
    activeStore |= other.activeStore;
    activeInit |= other.activeInit;
    activeEscape |= other.activeEscape;
    return ChangeResult::Change;
  }
};

class MemoryActivity : public AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  /// Clear all modifications.
  ChangeResult reset() {
    if (activityStates.empty())
      return ChangeResult::NoChange;
    activityStates.clear();
    return ChangeResult::Change;
  }

  bool hasActiveData(Value value) const {
    const auto &state = activityStates.lookup(value);
    return state.activeInit || state.activeStore;
  }

  bool activeDataFlowsOut(Value value) const {
    const auto &state = activityStates.lookup(value);
    return state.activeLoad || state.activeEscape;
  }

  /// Set the internal activity state.
  ChangeResult addAllocation(Value value) {
    if (activityStates.contains(value)) {
      return ChangeResult::NoChange;
    }
    activityStates.insert({value, MemoryActivityState{.activeLoad = false,
                                                      .activeStore = false,
                                                      .activeInit = false,
                                                      .activeEscape = false}});
    return ChangeResult::Change;
  }

  ChangeResult setActiveStore(Value value, bool activeStore) {
    ChangeResult result = ChangeResult::NoChange;
    auto &state = activityStates[value];
    if (state.activeStore != activeStore)
      result = ChangeResult::Change;
    state.activeStore = activeStore;
    return result;
  }

  ChangeResult setActiveLoad(Value value, bool activeLoad) {
    ChangeResult result = ChangeResult::NoChange;
    auto &state = activityStates[value];
    if (state.activeLoad != activeLoad) {
      result = ChangeResult::Change;
    }
    state.activeLoad = activeLoad;
    return result;
  }

  ChangeResult setActiveInit(Value value, bool activeInit) {
    auto &state = activityStates[value];
    ChangeResult result = ChangeResult::NoChange;
    if (state.activeInit != activeInit) {
      result = ChangeResult::Change;
      state.activeInit = activeInit;
    }
    return result;
  }

  ChangeResult setActiveEscape(Value value, bool activeEscape) {
    auto &state = activityStates[value];
    ChangeResult result = ChangeResult::NoChange;
    if (state.activeEscape != activeEscape) {
      result = ChangeResult::Change;
      state.activeEscape = activeEscape;
    }
    return result;
  }

  void print(raw_ostream &os) const override {
    if (activityStates.empty()) {
      os << "<memory activity state was empty>"
         << "\n";
    }
    for (const auto &[value, state] : activityStates) {
      os << value << ": load " << state.activeLoad << " store "
         << state.activeStore << " init " << state.activeInit << " escape "
         << state.activeEscape << "\n";
    }
  }

  raw_ostream &operator<<(raw_ostream &os) const {
    print(os);
    return os;
  }

protected:
  DenseMap<Value, MemoryActivityState> activityStates;
};

class ForwardMemoryActivity : public MemoryActivity {
public:
  using MemoryActivity::MemoryActivity;

  /// Join the activity states.
  ChangeResult join(const AbstractDenseLattice &lattice) {
    const auto &rhs = static_cast<const ForwardMemoryActivity &>(lattice);
    ChangeResult result = ChangeResult::NoChange;
    for (const auto &[value, rhsState] : rhs.activityStates) {
      auto &lhsState = activityStates[value];
      result |= lhsState.merge(rhsState);
    }
    return result;
  }
};

class BackwardMemoryActivity : public MemoryActivity {
public:
  using MemoryActivity::MemoryActivity;

  ChangeResult meet(const AbstractDenseLattice &lattice) override {
    const auto &rhs = static_cast<const BackwardMemoryActivity &>(lattice);
    ChangeResult result = ChangeResult::NoChange;
    for (const auto &[value, rhsState] : rhs.activityStates) {
      auto &lhsState = activityStates[value];
      result |= lhsState.merge(rhsState);
    }
    return result;
  }
};

/// Sparse activity analysis reasons about activity by traversing forward down
/// the def-use chains starting from active function arguments.
class SparseForwardActivityAnalysis
    : public SparseForwardDataFlowAnalysis<ForwardValueActivity> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// In general, we don't know anything about entry operands.
  void setToEntryState(ForwardValueActivity *lattice) override {
    // errs() << "sparse forward setting to entry state\n";
    propagateIfChanged(lattice, lattice->join(ValueActivity()));
  }

  void visitOperation(Operation *op,
                      ArrayRef<const ForwardValueActivity *> operands,
                      ArrayRef<ForwardValueActivity *> results) override {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      for (auto result : results) {
        result->join(ValueActivity::getConstant());
      }
      return;
    }

    // Bail out if this op affects memory.
    if (auto memory = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      memory.getEffects(effects);
      if (!effects.empty())
        return;
    }

    // For value-based AA, assume any active argument leads to an active
    // result.
    // TODO: Could prune values based on the types of the operands (but would
    // require type analysis for full robustness)
    // TODO: Could we differentiate between values that don't propagate active
    // information? memcpy, stores don't produce active results (they don't
    // produce any). There are undoubtedly also function calls that don't
    // produce active results.
    ValueActivity joinedResult;
    for (auto operand : operands) {
      joinedResult = ValueActivity::merge(joinedResult, operand->getValue());
    }

    for (auto result : results) {
      propagateIfChanged(result, result->join(joinedResult));
    }
  }
};

class SparseBackwardActivityAnalysis
    : public SparseBackwardDataFlowAnalysis<BackwardValueActivity> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void setToExitState(BackwardValueActivity *lattice) override {
    // errs() << "backward sparse setting to exit state\n";
  }

  void visitBranchOperand(OpOperand &operand) override {}

  void
  visitOperation(Operation *op, ArrayRef<BackwardValueActivity *> operands,
                 ArrayRef<const BackwardValueActivity *> results) override {
    // Bail out if the op propagates memory
    if (auto memory = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      memory.getEffects(effects);
      if (!effects.empty())
        return;
    }

    // Propagate all operands to all results
    for (auto operand : operands) {
      if (Operation *definingOp = operand->getPoint().getDefiningOp()) {
        if (definingOp->hasTrait<OpTrait::ConstantLike>()) {
          propagateIfChanged(operand,
                             operand->meet(ValueActivity::getConstant()));
          continue;
        }
      }
      for (auto result : results) {
        meet(operand, *result);
      }
    }
  }
};

// When applying a transfer function to a store from memory, we need to know
// what value is being stored.
std::optional<Value> getStored(Operation *op) {
  if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
    return storeOp.getValue();
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return storeOp.getValue();
  }
  return std::nullopt;
}

std::optional<Value> getCopySource(Operation *op) {
  if (auto copyOp = dyn_cast<CopyOpInterface>(op)) {
    return copyOp.getSource();
  } else if (isa<LLVM::MemcpyOp, LLVM::MemcpyInlineOp, LLVM::MemmoveOp>(op)) {
    return op->getOperand(1);
  }
  return std::nullopt;
}

/// The dense analyses operate using a pointer's "canonical allocation", the
/// Value corresponding to its allocation.
void forEachAliasedAlloc(const AliasClassLattice *ptrAliasClass,
                         const AvailableAllocations *availableAllocs,
                         ValueRange entryAllocs,
                         function_ref<void(Value alloc)> forEachFn) {
  SmallVector<Value> canonicalAllocs;
  if (ptrAliasClass->isEntry()) {
    assert(!entryAllocs.empty() &&
           "found an entry alias class but entry args was empty");
    // Take the first unannotated entry argument as the canonical allocation
    canonicalAllocs.push_back(entryAllocs.front());
  } else if (ptrAliasClass->isUnknown()) {
    // Unknown pointers alias with the unknown entry arguments and all
    // known allocations
    if (!entryAllocs.empty())
      canonicalAllocs.push_back(entryAllocs.front());
    availableAllocs->getAllocations(canonicalAllocs);
  } else {
    ptrAliasClass->getCanonicalAllocations(canonicalAllocs);
  }

  for (Value alloc : canonicalAllocs) {
    forEachFn(alloc);
  }
}

class DenseForwardActivityAnalysis
    : public DenseForwardDataFlowAnalysis<ForwardMemoryActivity> {
public:
  DenseForwardActivityAnalysis(DataFlowSolver &solver,
                               SmallVectorImpl<Value> &entryAllocs)
      : DenseForwardDataFlowAnalysis(solver), entryAllocs(entryAllocs) {}

  void visitOperation(Operation *op, const ForwardMemoryActivity &before,
                      ForwardMemoryActivity *after) override {
    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    // If we can't reason about the memory effects, then conservatively assume
    // we can't deduce anything about activity via side-effects.
    if (!memory)
      return setToEntryState(after);

    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);

    ChangeResult result = after->join(before);
    for (const auto &effect : effects) {
      Value value = effect.getValue();

      // If we see an effect on anything other than a value, assume we can't
      // deduce anything about the activity.
      if (!value)
        return setToEntryState(after);

      // In forward-flow, a value is active if loaded from a memory resource
      // that has previously been actively stored to.
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
        auto *availableAllocs = getOrCreateFor<AvailableAllocations>(op, op);
        forEachAliasedAlloc(
            ptrAliasClass, availableAllocs, entryAllocs, [&](Value alloc) {
              if (before.hasActiveData(alloc)) {
                result |= after->setActiveLoad(alloc, true);
                for (OpResult opResult : op->getResults()) {
                  // Mark the result as (forward) active
                  // TODO: We might need type analysis here
                  ValueActivity resultActivity =
                      isa<FloatType, ComplexType>(opResult.getType())
                          ? ValueActivity::getActiveVal()
                          : ValueActivity::getActivePtr();
                  auto *valueState =
                      getOrCreate<ForwardValueActivity>(opResult);
                  propagateIfChanged(valueState,
                                     valueState->join(resultActivity));
                }

                if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
                  result |= after->setActiveLoad(alloc, true);

                  // propagate from input to block argument
                  for (OpOperand *inputOperand :
                       linalgOp.getDpsInputOperands()) {
                    if (inputOperand->get() == value) {
                      auto *valueState = getOrCreate<ForwardValueActivity>(
                          linalgOp.getMatchingBlockArgument(inputOperand));
                      propagateIfChanged(
                          valueState,
                          valueState->join(ValueActivity::getActiveVal()));
                    }
                  }
                }
              }
            });
      }

      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        std::optional<Value> stored = getStored(op);
        if (stored.has_value()) {
          auto *valueState = getOrCreateFor<ForwardValueActivity>(op, *stored);
          if (valueState->getValue().isActiveVal() ||
              valueState->getValue().isActivePtr()) {
            auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
            auto *availableAllocs =
                getOrCreateFor<AvailableAllocations>(op, op);
            forEachAliasedAlloc(
                ptrAliasClass, availableAllocs, entryAllocs, [&](Value alloc) {
                  // Mark the pointer as having been actively stored into
                  result |= after->setActiveStore(alloc, true);

                  // Mark the destination pointer as an active pointer
                  auto ptrValueActivity =
                      getOrCreate<ForwardValueActivity>(alloc);
                  propagateIfChanged(
                      ptrValueActivity,
                      ptrValueActivity->join(ValueActivity::getActivePtr()));
                });
          }
        } else if (auto copySource = getCopySource(op)) {
          auto *srcAliasClass =
              getOrCreateFor<AliasClassLattice>(op, *copySource);
          auto *availableAllocs = getOrCreateFor<AvailableAllocations>(op, op);
          // Do we need to iterate over aliased allocations here?
          // If there's *any* aliased allocation that contains active data
          forEachAliasedAlloc(
              srcAliasClass, availableAllocs, entryAllocs, [&](Value srcAlloc) {
                if (before.hasActiveData(srcAlloc)) {
                  auto *destAliasClass =
                      getOrCreateFor<AliasClassLattice>(op, value);
                  forEachAliasedAlloc(
                      destAliasClass, availableAllocs, entryAllocs,
                      [&](Value destAlloc) {
                        result |= after->setActiveStore(destAlloc, true);
                        auto ptrValueActivity =
                            getOrCreate<ForwardValueActivity>(destAlloc);
                        propagateIfChanged(ptrValueActivity,
                                           ptrValueActivity->join(
                                               ValueActivity::getActivePtr()));
                      });
                }
              });
        } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
          // linalg.yield stores to the corresponding value.
          for (OpOperand *dpsInit : linalgOp.getDpsInitOperands()) {
            if (dpsInit->get() == value) {
              int64_t resultIndex =
                  dpsInit->getOperandNumber() - linalgOp.getNumDpsInputs();
              Value yieldOperand =
                  linalgOp.getBlock()->getTerminator()->getOperand(resultIndex);
              auto *valueState =
                  getOrCreateFor<ForwardValueActivity>(op, yieldOperand);
              if (valueState->getValue().isActiveVal()) {
                auto *ptrAliasClass =
                    getOrCreateFor<AliasClassLattice>(op, value);
                auto *availableAllocs =
                    getOrCreateFor<AvailableAllocations>(op, op);
                forEachAliasedAlloc(
                    ptrAliasClass, availableAllocs, entryAllocs,
                    [&](Value alloc) {
                      result |= after->setActiveStore(alloc, true);
                      auto ptrValueActivity =
                          getOrCreate<ForwardValueActivity>(alloc);
                      propagateIfChanged(ptrValueActivity,
                                         ptrValueActivity->join(
                                             ValueActivity::getActivePtr()));
                    });
              }
            }
          }
        }
      }
    }
    propagateIfChanged(after, result);
  }

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const ForwardMemoryActivity &before,
                                    ForwardMemoryActivity *after) override {
    join(after, before);
  }

  /// Entry states in the framework are synonymously with pessimistic states,
  /// which in dense AA would be "everything is active". However, dense AA is
  /// actually initialized optimistically, assuming "everything is constant".
  void setToEntryState(ForwardMemoryActivity *lattice) override {}

private:
  // The entry arguments for the top-level function being differentiated
  // without alias attributes.
  SmallVectorImpl<Value> &entryAllocs;
};

class DenseBackwardActivityAnalysis
    : public DenseBackwardDataFlowAnalysis<BackwardMemoryActivity> {
public:
  DenseBackwardActivityAnalysis(DataFlowSolver &solver,
                                SymbolTableCollection &symbolTable,
                                SmallVectorImpl<Value> &entryAllocs)
      : DenseBackwardDataFlowAnalysis(solver, symbolTable),
        entryAllocs(entryAllocs) {}

  void visitOperation(Operation *op, const BackwardMemoryActivity &after,
                      BackwardMemoryActivity *before) override {
    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    // If we can't reason about the memory effects, then conservatively assume
    // we can't deduce anything about activity via side-effects.
    if (!memory)
      return setToExitState(before);

    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);

    ChangeResult result = before->meet(after);
    for (const auto &effect : effects) {
      Value value = effect.getValue();

      // If we see an effect on anything other than a value, assume we can't
      // deduce anything about the activity.
      if (!value)
        return setToExitState(before);

      // In backward-flow, a value is active if stored into a memory resource
      // that has subsequently been actively loaded from.
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        for (Value opResult : op->getResults()) {
          auto *valueState =
              getOrCreateFor<BackwardValueActivity>(op, opResult);
          if (valueState->getValue().isActiveVal() ||
              valueState->getValue().isActivePtr()) {
            auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
            auto *availableAllocs =
                getOrCreateFor<AvailableAllocations>(op, op);
            forEachAliasedAlloc(
                ptrAliasClass, availableAllocs, entryAllocs, [&](Value alloc) {
                  result |= before->setActiveLoad(alloc, true);

                  auto ptrState = getOrCreate<BackwardValueActivity>(value);
                  propagateIfChanged(
                      ptrState, ptrState->meet(ValueActivity::getActivePtr()));
                });
          }
        }
      }
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
        auto *availableAllocs = getOrCreateFor<AvailableAllocations>(op, op);
        std::optional<Value> stored = getStored(op);
        std::optional<Value> copySource = getCopySource(op);
        forEachAliasedAlloc(
            ptrAliasClass, availableAllocs, entryAllocs, [&](Value alloc) {
              if (stored.has_value() && after.activeDataFlowsOut(alloc)) {
                result |= before->setActiveStore(alloc, true);
                ValueActivity resultActivity =
                    isa<FloatType, ComplexType>(stored->getType())
                        ? ValueActivity::getActiveVal()
                        : ValueActivity::getActivePtr();
                auto *valueState = getOrCreate<BackwardValueActivity>(*stored);
                propagateIfChanged(valueState,
                                   valueState->meet(resultActivity));
              } else if (copySource.has_value() &&
                         after.activeDataFlowsOut(alloc)) {
                result |= before->setActiveStore(alloc, true);
                auto *srcAliasClass =
                    getOrCreateFor<AliasClassLattice>(op, *copySource);
                forEachAliasedAlloc(
                    srcAliasClass, availableAllocs, entryAllocs,
                    [&](Value srcAlloc) {
                      before->setActiveLoad(srcAlloc, true);
                      auto *valueState =
                          getOrCreate<BackwardValueActivity>(srcAlloc);
                      propagateIfChanged(
                          valueState,
                          valueState->meet(ValueActivity::getActivePtr()));
                    });
              } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
                if (after.activeDataFlowsOut(alloc)) {
                  result |= before->setActiveStore(alloc, true);
                  for (OpOperand *dpsInit : linalgOp.getDpsInitOperands()) {
                    if (dpsInit->get() == value) {
                      int64_t resultIndex = dpsInit->getOperandNumber() -
                                            linalgOp.getNumDpsInputs();
                      Value yieldOperand =
                          linalgOp.getBlock()->getTerminator()->getOperand(
                              resultIndex);
                      auto *valueState =
                          getOrCreate<BackwardValueActivity>(yieldOperand);
                      propagateIfChanged(
                          valueState,
                          valueState->meet(ValueActivity::getActiveVal()));
                    }
                  }
                }
              }
            });
      }
    }
    propagateIfChanged(before, result);
  }

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const BackwardMemoryActivity &after,
                                    BackwardMemoryActivity *before) override {
    meet(before, after);
  }

  void setToExitState(BackwardMemoryActivity *lattice) override {}

private:
  SmallVectorImpl<Value> &entryAllocs;
};

void enzyme::runDataFlowActivityAnalysis(
    FunctionOpInterface callee, ArrayRef<enzyme::Activity> argumentActivity,
    bool print, bool verbose) {
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  // Keep track of all the entry pointers without noalias attributes (that
  // may alias all other entry pointers)
  SmallVector<Value> entryAllocations;

  solver.load<AvailableAllocAnalysis>();
  solver.load<enzyme::AliasAnalysis>();
  solver.load<SparseForwardActivityAnalysis>();
  solver.load<DenseForwardActivityAnalysis>(entryAllocations);
  solver.load<SparseBackwardActivityAnalysis>(symbolTable);
  solver.load<DenseBackwardActivityAnalysis>(symbolTable, entryAllocations);

  // Required for the dataflow framework to traverse region-based control flow
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();

  // Initialize the argument states based on the given activity annotations.
  auto *noaliasEntryAllocs = solver.getOrCreateState<AvailableAllocations>(
      &callee.getFunctionBody().front());
  auto *initialState = solver.getOrCreateState<ForwardMemoryActivity>(
      &callee.getFunctionBody().front());
  for (const auto &[arg, activity] :
       llvm::zip(callee.getArguments(), argumentActivity)) {
    // Need to determine if this is a pointer (or memref) or not, the dup
    // activity is kind of a proxy
    if (activity == enzyme::Activity::enzyme_dup ||
        activity == enzyme::Activity::enzyme_dupnoneed) {
      initialState->setActiveInit(arg, true);

      if (callee.getArgAttr(arg.getArgNumber(),
                            LLVM::LLVMDialect::getNoAliasAttrName()))
        noaliasEntryAllocs->addAllocation(arg);
      else
        entryAllocations.push_back(arg);

      auto *argLattice = solver.getOrCreateState<ForwardValueActivity>(arg);
      argLattice->join(ValueActivity::getActivePtr());
    } else {
      auto *argLattice = solver.getOrCreateState<ForwardValueActivity>(arg);
      auto state = activity == enzyme::Activity::enzyme_const
                       ? ValueActivity::getConstant()
                       : ValueActivity::getActiveVal();
      argLattice->join(state);
    }
  }

  // Detect function returns as direct children of the FunctionOpInterface
  // that have the ReturnLike trait.
  SmallPtrSet<Operation *, 2> returnOps;
  for (Operation &op : callee.getFunctionBody().getOps()) {
    if (op.hasTrait<OpTrait::ReturnLike>()) {
      returnOps.insert(&op);
      auto *returnDenseLattice =
          solver.getOrCreateState<BackwardMemoryActivity>(&op);
      for (const auto &[arg, activity] :
           llvm::zip(callee.getArguments(), argumentActivity)) {
        if (activity == enzyme::Activity::enzyme_dup ||
            activity == enzyme::Activity::enzyme_dupnoneed) {
          returnDenseLattice->setActiveEscape(arg, true);
          auto *argLattice =
              solver.getOrCreateState<BackwardValueActivity>(arg);
          argLattice->meet(ValueActivity::getActivePtr());
        }
      }
      for (Value operand : op.getOperands()) {
        auto *returnLattice =
            solver.getOrCreateState<BackwardValueActivity>(operand);
        // Very basic type inference of the type
        if (isa<FloatType>(operand.getType())) {
          returnLattice->meet(ValueActivity::getActiveVal());
        } else if (isa<MemRefType, LLVM::LLVMPointerType>(operand.getType())) {
          returnLattice->meet(ValueActivity::getActivePtr());
        } else {
          returnLattice->meet(ValueActivity::getConstant());
        }
      }
    }
  }

  if (failed(solver.initializeAndRun(callee->getParentOfType<ModuleOp>()))) {
    assert(false && "dataflow analysis failed\n");
  }

  if (print) {
    auto isConstantValue = [&](DataFlowSolver &solver, Value value) {
      // TODO: integers/vectors that might be pointers
      // if (isa<LLVM::LLVMPointerType, MemRefType>(value.getType())) {
      //   assert(returnOps.size() == 1);
      //   auto *fma =
      //       solver.lookupState<ForwardMemoryActivity>(*returnOps.begin());
      //   auto *bma = solver.lookupState<BackwardMemoryActivity>(
      //       &callee.getFunctionBody().front().front());

      //   auto *aliasClass = solver.lookupState<AliasClassLattice>(value);
      //   if (aliasClass->isEntry || aliasClass->isUnknown) {
      //     errs() << "[activity analysis] unimplemented entry or unknown";
      //   }
      //   SmallVector<Value> canonicalAllocs;
      //   aliasClass->getCanonicalAllocations(canonicalAllocs);
      //   bool activePtr = false;
      //   for (Value alloc : canonicalAllocs) {
      //     // How do we combine the dense states correctly?
      //     // If it's forward load and backward store it's definitely active
      //     activePtr |=
      //         fma->hasActiveData(alloc) && bma->activeDataFlowsOut(alloc);
      //   }
      //   return !activePtr;
      // }

      auto fva = solver.lookupState<ForwardValueActivity>(value);
      auto bva = solver.lookupState<BackwardValueActivity>(value);
      // bool forwardActive = fva && fva->getValue().isActiveVal();
      // bool backwardActive = bva && bva->getValue().isActiveVal();
      bool forwardActive = fva && (fva->getValue().isActiveVal() ||
                                   fva->getValue().isActivePtr());
      bool backwardActive = bva && (bva->getValue().isActiveVal() ||
                                    bva->getValue().isActivePtr());
      if (forwardActive && backwardActive)
        return false;
      return true;
    };

    errs() << FlatSymbolRefAttr::get(callee) << ":\n";
    for (BlockArgument arg : callee.getArguments()) {
      if (Attribute tagAttr =
              callee.getArgAttr(arg.getArgNumber(), "enzyme.tag")) {
        errs() << "  " << tagAttr << ": "
               << (isConstantValue(solver, arg) ? "Constant" : "Active")
               << "\n";
      }
    }
    callee.walk([&](Operation *op) {
      if (op->hasAttr("tag")) {
        errs() << "  " << op->getAttr("tag") << ": ";
        for (OpResult opResult : op->getResults()) {
          errs() << (isConstantValue(solver, opResult) ? "Constant" : "Active")
                 << "\n";
        }
      }
      if (verbose) {
        // Annotate each op's results with its value activity states
        for (OpResult result : op->getResults()) {
          auto forwardValueActivity =
              solver.lookupState<ForwardValueActivity>(result);
          if (forwardValueActivity) {
            std::string dest, key{"fva"};
            llvm::raw_string_ostream os(dest);
            if (op->getNumResults() != 1)
              key += result.getResultNumber();
            forwardValueActivity->getValue().print(os);
            op->setAttr(key, StringAttr::get(op->getContext(), dest));
          }

          auto backwardValueActivity =
              solver.lookupState<BackwardValueActivity>(result);
          if (backwardValueActivity) {
            std::string dest, key{"bva"};
            llvm::raw_string_ostream os(dest);
            if (op->getNumResults() != 1)
              key += result.getResultNumber();
            backwardValueActivity->getValue().print(os);
            op->setAttr(key, StringAttr::get(op->getContext(), dest));
          }
        }
      }
    });

    if (verbose) {
      // Annotate function attributes
      for (BlockArgument arg : callee.getArguments()) {
        auto backwardValueActivity =
            solver.lookupState<BackwardValueActivity>(arg);
        if (backwardValueActivity) {
          std::string dest;
          llvm::raw_string_ostream os(dest);
          backwardValueActivity->getValue().print(os);
          callee.setArgAttr(arg.getArgNumber(), "enzyme.bva",
                            StringAttr::get(callee->getContext(), dest));
        }
      }

      for (Operation *returnOp : returnOps) {
        auto *allocations = solver.lookupState<AvailableAllocations>(returnOp);
        errs() << "available allocations at end of function: " << *allocations
               << "\n";

        auto *state = solver.lookupState<ForwardMemoryActivity>(returnOp);
        if (state)
          errs() << "resulting forward state:\n" << *state << "\n";
        else
          errs() << "state was null\n";
      }

      auto startState = solver.lookupState<BackwardMemoryActivity>(
          &callee.getFunctionBody().front().front());
      if (startState)
        errs() << "backwards end state:\n" << *startState << "\n";
      else
        errs() << "backwards end state was null\n";
    }
  }
}
