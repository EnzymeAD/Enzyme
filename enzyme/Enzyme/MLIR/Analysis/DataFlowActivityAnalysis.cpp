#include "DataFlowActivityAnalysis.h"
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

/// From Enzyme proper's activity analysis, there are four activity states.
// constant instruction vs constant value, a value/instruction (one and the same
// in LLVM) can be a constant instruction but active value, active instruction
// but constant value, or active/constant both.

// In MLIR, values are not the same as instructions. Many operations produce
// zero or one result, but there are operations that can produce multiple.

// The result of activity states are potentially different for multiple
// enzyme.autodiff calls.

// We could use enyzme::Activity here but I don't know that it would help from a
// dataflow perspective (distinguishing between enzyme_dup, enzyme_dupnoneed,
// enzyme_out, which are all active)
enum class ActivityKind { Active, Constant };

using llvm::errs;
class ValueActivity {
public:
  static ValueActivity getConstant() {
    return ValueActivity(ActivityKind::Constant);
  }

  static ValueActivity getActive() {
    return ValueActivity(ActivityKind::Active);
  }

  bool isActive() const {
    return value.has_value() && *value == ActivityKind::Active;
  }

  bool isConstant() const {
    return value.has_value() && *value == ActivityKind::Constant;
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

  static ValueActivity join(const ValueActivity &lhs,
                            const ValueActivity &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs.isConstant() && rhs.isConstant()) {
      return ValueActivity::getConstant();
    }

    return ValueActivity::getActive();
  }

  static ValueActivity meet(const ValueActivity &lhs,
                            const ValueActivity &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs.isConstant() && rhs.isConstant()) {
      return ValueActivity::getConstant();
    }

    return ValueActivity::getActive();
  }

  void print(raw_ostream &os) const {
    if (!value) {
      os << "<uninitialized>";
      return;
    }
    switch (*value) {
    case ActivityKind::Active:
      os << "Active";
      break;
    case ActivityKind::Constant:
      os << "Constant";
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

// Inheriting Lattice<ValueActivity> would be the easiest (because we define a
// meet function for ValueActivity) but the meet function doesn't look like it's
// being picked up for some reason. I don't know how to debug the trait that's
// causing this issue.
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
    auto met = ValueActivity::meet(getValue(), other);
    if (getValue() == met) {
      return ChangeResult::NoChange;
    }

    value = met;
    return ChangeResult::Change;
  }

private:
  ValueActivity value;
};

/// This needs to keep track of three things:
///   1. Could active info store in?
///   2. Could active info load out?
///   3. Could constant info propagate (store?) in?
///
/// Active: active in && active out && !const in
/// ActiveOrConstant: active in && active out && const in
/// Constant: everything else
struct MemoryActivityState {
  bool activeLoad;
  bool activeStore;
  // Active init is like active store, but a special case for arguments. We need
  // to distinguish arguments that start with active data vs arguments that get
  // active data stored into them during the function.
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

  bool mayAlias(Value lhs, Value rhs) const {
    return !const_cast<LocalAliasAnalysis *>(&aliasAnalysis)
                ->alias(lhs, rhs)
                .isNo();
  }

  bool hasActiveData(Value value) const {
    const auto &state = activityStates.lookup(value);
    if (state.activeInit || state.activeStore) {
      return true;
    }

    bool activeData = false;
    for (const auto &[other, state] : activityStates)
      if (mayAlias(value, other)) {
        activeData |= state.activeStore || state.activeInit;
        if (activeData)
          return activeData;
      }

    return activeData;
  }

  bool activeDataFlowsOut(Value value) const {
    // const auto &state = activityStates.lookup(value);
    // return state.activeLoad || state.activeEscape;
    bool flowsOut = false;
    for (const auto &[other, state] : activityStates)
      if (mayAlias(value, other)) {
        flowsOut |= state.activeLoad || state.activeEscape;
        if (flowsOut)
          return flowsOut;
      }

    return flowsOut;
  }

  /// Set the internal activity state.
  ChangeResult setActiveStore(Value value, bool activeStore) {
    // Make sure an entry for the value exists
    activityStates[value];
    ChangeResult result = ChangeResult::NoChange;
    for (auto &[other, state] : activityStates) {
      if (mayAlias(value, other) && state.activeStore != activeStore) {
        result |= ChangeResult::Change;
        state.activeStore = activeStore;
      }
    }
    return result;
  }

  ChangeResult setActiveLoad(Value value, bool activeLoad) {
    activityStates[value];
    ChangeResult result = ChangeResult::NoChange;
    for (auto &[other, state] : activityStates) {
      if (mayAlias(value, other) && state.activeLoad != activeLoad) {
        result = ChangeResult::Change;
        state.activeLoad = activeLoad;
      }
    }
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
  LocalAliasAnalysis aliasAnalysis;
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
      if (lhsState != rhsState) {
        lhsState.activeLoad |= rhsState.activeLoad;
        lhsState.activeStore |= rhsState.activeStore;
        lhsState.activeInit |= rhsState.activeInit;
        lhsState.activeEscape |= rhsState.activeEscape;
        result |= ChangeResult::Change;
      }
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
      if (lhsState != rhsState) {
        lhsState.activeLoad |= rhsState.activeLoad;
        lhsState.activeStore |= rhsState.activeStore;
        lhsState.activeInit |= rhsState.activeInit;
        lhsState.activeEscape |= rhsState.activeEscape;
        result |= ChangeResult::Change;
      }
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

    // For value-based AA, assume any active argument leads to an active result.
    // TODO: Could prune values based on the types of the operands (but would
    // require type analysis for full robustness)
    // TODO: Could we differentiate between values that don't propagate active
    // information? memcpy, stores don't produce active results (they don't
    // produce any). There are undoubtedly also function calls that don't
    // produce active results.
    ValueActivity joinedResult;
    for (auto operand : operands) {
      joinedResult = ValueActivity::join(joinedResult, operand->getValue());
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
    errs() << "setting to exit state\n";
  }

  void visitBranchOperand(OpOperand &operand) override {}

  void
  visitOperation(Operation *op, ArrayRef<BackwardValueActivity *> operands,
                 ArrayRef<const BackwardValueActivity *> results) override {
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

std::optional<Value> getStored(Operation *op) {
  if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
    return storeOp.getValue();
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return storeOp.getValue();
  }
  return std::nullopt;
}

class DenseForwardActivityAnalysis
    : public DenseForwardDataFlowAnalysis<ForwardMemoryActivity> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

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

      // TODO: From the upstream test dense analysis, we may need to copy paste
      // "Underlying Value" analysis to traverse call graphs correctly.

      // value =
      // getMostUnderlyingValue(value, [&](Value value) {
      //   return getOrCreateFor<UnderlyingValueLattice>(op, value);
      // });
      // if (!value)
      //   return;

      // In forward-flow, a value is active if loaded from a memory resource
      // that has previously been actively stored to.
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        if (before.hasActiveData(value)) {
          result |= after->setActiveLoad(value, true);
          for (OpResult opResult : op->getResults()) {
            // Mark the result as (forward) active
            auto *valueState = getOrCreate<ForwardValueActivity>(opResult);
            propagateIfChanged(valueState,
                               valueState->join(ValueActivity::getActive()));
          }
          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            result |= after->setActiveLoad(value, true);

            // propagate from input to block argument
            for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
              if (inputOperand->get() == value) {
                auto *valueState = getOrCreate<ForwardValueActivity>(
                    linalgOp.getMatchingBlockArgument(inputOperand));
                propagateIfChanged(
                    valueState, valueState->join(ValueActivity::getActive()));
              }
            }
          }
        }
      }

      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        std::optional<Value> stored = getStored(op);
        if (stored.has_value()) {
          auto *valueState = getOrCreateFor<ForwardValueActivity>(op, *stored);
          if (valueState->getValue().isActive()) {
            result |= after->setActiveStore(value, true);

            auto ptrValueActivity = getOrCreate<ForwardValueActivity>(value);
            propagateIfChanged(
                ptrValueActivity,
                ptrValueActivity->join(ValueActivity::getActive()));
          }
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
              if (valueState->getValue().isActive()) {
                result |= after->setActiveStore(value, true);
              }
            }
          }
        }
      }
    }
    propagateIfChanged(after, result);
  }

  // Not sure what this should be, unknown?
  void setToEntryState(ForwardMemoryActivity *lattice) override {
    errs() << "setting to entry state\n";
    propagateIfChanged(lattice, lattice->reset());
  }
};

class DenseBackwardActivityAnalysis
    : public DenseBackwardDataFlowAnalysis<BackwardMemoryActivity> {
public:
  using DenseBackwardDataFlowAnalysis::DenseBackwardDataFlowAnalysis;

  void visitOperation(Operation *op, const BackwardMemoryActivity &after,
                      BackwardMemoryActivity *before) override {
    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    ChangeResult result = before->meet(after);
    // If we can't reason about the memory effects, then conservatively assume
    // we can't deduce anything about activity via side-effects.
    if (!memory)
      return setToExitState(before);

    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);

    for (const auto &effect : effects) {
      Value value = effect.getValue();

      // If we see an effect on anything other than a value, assume we can't
      // deduce anything about the activity.
      if (!value)
        return setToExitState(before);

      // TODO: From the upstream test dense analysis, we may need to copy paste
      // "Underlying Value" analysis to traverse call graphs correctly.

      // value =
      // getMostUnderlyingValue(value, [&](Value value) {
      //   return getOrCreateFor<UnderlyingValueLattice>(op, value);
      // });
      // if (!value)
      //   return;

      // In backward-flow, a value is active if stored into a memory resource
      // that has subsequently been actively loaded from.
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        for (Value opResult : op->getResults()) {
          auto *valueState =
              getOrCreateFor<BackwardValueActivity>(op, opResult);
          if (valueState->getValue().isActive()) {
            result |= before->setActiveLoad(value, true);
          }
        }
      }
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        std::optional<Value> stored = getStored(op);
        if (stored.has_value()) {
          if (after.activeDataFlowsOut(value)) {
            result |= before->setActiveStore(value, true);
            auto *valueState = getOrCreate<BackwardValueActivity>(*stored);
            propagateIfChanged(valueState,
                               valueState->meet(ValueActivity::getActive()));
          }
        } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
          if (after.activeDataFlowsOut(value)) {
            result |= before->setActiveStore(value, true);
            for (OpOperand *dpsInit : linalgOp.getDpsInitOperands()) {
              if (dpsInit->get() == value) {
                int64_t resultIndex =
                    dpsInit->getOperandNumber() - linalgOp.getNumDpsInputs();
                Value yieldOperand =
                    linalgOp.getBlock()->getTerminator()->getOperand(
                        resultIndex);
                auto *valueState =
                    getOrCreate<BackwardValueActivity>(yieldOperand);
                propagateIfChanged(
                    valueState, valueState->meet(ValueActivity::getActive()));
              }
            }
          }
        }
      }
    }
    propagateIfChanged(before, result);
  }

  void setToExitState(BackwardMemoryActivity *lattice) override {
    errs() << "setting to exit state\n";
    propagateIfChanged(lattice, lattice->reset());
  }
};

void enzyme::runDataFlowActivityAnalysis(
    FunctionOpInterface callee, ArrayRef<enzyme::Activity> argumentActivity,
    bool print, bool verbose) {
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;

  solver.load<SparseForwardActivityAnalysis>();
  solver.load<DenseForwardActivityAnalysis>();
  solver.load<SparseBackwardActivityAnalysis>(symbolTable);
  solver.load<DenseBackwardActivityAnalysis>(symbolTable);

  // Required for the dataflow framework to traverse region-based control flow
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();

  // Initialize the argument states based on the given activity annotations.
  for (const auto &[arg, activity] :
       llvm::zip(callee.getArguments(), argumentActivity)) {
    // Need to determine if this is a pointer (or memref) or not, the dup
    // activity is kind of a proxy
    if (activity == enzyme::Activity::enzyme_dup ||
        activity == enzyme::Activity::enzyme_dupnoneed) {
      auto *initialState = solver.getOrCreateState<ForwardMemoryActivity>(
          &callee.getFunctionBody().front());
      initialState->setActiveInit(arg, true);

      // May be too conservative to mark the duplicated arguments as active
      // values.
      auto *argLattice = solver.getOrCreateState<ForwardValueActivity>(arg);
      auto state = activity == enzyme::Activity::enzyme_const
                       ? ValueActivity::getConstant()
                       : ValueActivity::getActive();
      argLattice->join(state);

      auto *backwardLattice =
          solver.getOrCreateState<BackwardValueActivity>(arg);
      backwardLattice->meet(state);
    } else {
      auto *argLattice = solver.getOrCreateState<ForwardValueActivity>(arg);
      auto state = activity == enzyme::Activity::enzyme_const
                       ? ValueActivity::getConstant()
                       : ValueActivity::getActive();
      argLattice->join(state);
    }
  }

  // TODO: Double-check the way we detect return-like ops. For now, all direct
  // children of the FunctionOpInterface that have the ReturnLike trait are
  // considered returns of that function. Other terminators (various
  // scf/affine/linalg yield) also have the ReturnLike trait, but nested regions
  // shouldn't be traversed.
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
        }
      }
      for (Value operand : op.getOperands()) {
        auto *returnLattice =
            solver.getOrCreateState<BackwardValueActivity>(operand);
        // Very basic type inference of the type
        returnLattice->meet(
            isa<FloatType, MemRefType, LLVM::LLVMPointerType>(operand.getType())
                ? ValueActivity::getActive()
                : ValueActivity::getConstant());
      }
    }
  }

  if (failed(solver.initializeAndRun(callee))) {
    assert(false && "dataflow analysis failed\n");
  }

  if (print) {
    errs() << FlatSymbolRefAttr::get(callee) << ":\n";
    for (BlockArgument arg : callee.getArguments()) {
      if (Attribute tagAttr =
              callee.getArgAttr(arg.getArgNumber(), "enzyme.tag")) {
        errs() << "  " << tagAttr << ": ";
        auto fva = solver.lookupState<ForwardValueActivity>(arg);
        auto bva = solver.lookupState<BackwardValueActivity>(arg);
        if (fva->getValue().isActive() && bva->getValue().isActive()) {
          errs() << "Active\n";
        } else {
          errs() << "Constant\n";
        }
      }
    }
    callee.walk([&](Operation *op) {
      if (op->hasAttr("tag")) {
        errs() << "  " << op->getAttr("tag") << ": ";
        for (OpResult opResult : op->getResults()) {
          auto fva = solver.lookupState<ForwardValueActivity>(opResult);
          auto bva = solver.lookupState<BackwardValueActivity>(opResult);
          if (fva->getValue().isActive() && bva->getValue().isActive()) {
            errs() << "Active\n";
          } else {
            errs() << "Constant\n";
          }
        }
      }

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
    });

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

    if (verbose) {
      auto startState = solver.lookupState<BackwardMemoryActivity>(
          &callee.getFunctionBody().front().front());
      if (startState)
        errs() << "backwards end state:\n" << *startState << "\n";
      else
        errs() << "backwards end state was null\n";

      for (Operation *returnOp : returnOps) {
        auto state = solver.lookupState<ForwardMemoryActivity>(returnOp);
        if (state)
          errs() << "resulting forward state:\n" << *state << "\n";
        else
          errs() << "state was null\n";
      }
    }
  }
}
