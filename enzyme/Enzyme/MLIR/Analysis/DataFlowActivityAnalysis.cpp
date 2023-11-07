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

enum class ActivityKind { Constant, ActiveVal, Unknown };

using llvm::errs;
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

  void print(raw_ostream &os) const {
    switch (value) {
    case ActivityKind::ActiveVal:
      os << "ActiveVal";
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
  /// The activity kind. Optimistically initialized to constant.
  ActivityKind value = ActivityKind::Constant;
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

  ChangeResult merge(const MemoryActivityState &other) {
    if (*this == other) {
      return ChangeResult::NoChange;
    }

    activeIn |= other.activeIn;
    activeOut |= other.activeOut;
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

  bool hasActiveData(DistinctAttr aliasClass) const {
    const auto &state = activityStates.lookup(aliasClass);
    return state.activeIn;
  }

  bool activeDataFlowsOut(DistinctAttr aliasClass) const {
    const auto &state = activityStates.lookup(aliasClass);
    return state.activeOut;
  }

  /// Set the internal activity state.
  ChangeResult setActiveIn(DistinctAttr aliasClass) {
    auto &state = activityStates[aliasClass];
    ChangeResult result =
        state.activeIn ? ChangeResult::NoChange : ChangeResult::Change;
    state.activeIn = true;
    return result;
  }

  ChangeResult setActiveOut(DistinctAttr aliasClass) {
    auto &state = activityStates[aliasClass];
    ChangeResult result =
        state.activeOut ? ChangeResult::NoChange : ChangeResult::Change;
    state.activeOut = true;
    return result;
  }

  void print(raw_ostream &os) const override {
    if (activityStates.empty()) {
      os << "<memory activity state was empty>"
         << "\n";
    }
    for (const auto &[value, state] : activityStates) {
      os << value << ": in " << state.activeIn << " out " << state.activeOut
         << "\n";
    }
  }

  raw_ostream &operator<<(raw_ostream &os) const {
    print(os);
    return os;
  }

protected:
  DenseMap<DistinctAttr, MemoryActivityState> activityStates;
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
    if (op->hasTrait<OpTrait::ConstantLike>())
      return;

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

  void visitCallOperand(OpOperand &operand) override {}

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
                         function_ref<void(DistinctAttr alloc)> forEachFn) {
  if (ptrAliasClass->isUnknown()) {
    // Unknown pointers alias with the unknown entry arguments and all
    // known allocations
    errs() << "unhandled unknown alias class\n";
  }
  for (DistinctAttr alloc : ptrAliasClass->getAliasClasses())
    forEachFn(alloc);
}

class DenseForwardActivityAnalysis
    : public DenseForwardDataFlowAnalysis<ForwardMemoryActivity> {
public:
  DenseForwardActivityAnalysis(DataFlowSolver &solver, Block *entryBlock,
                               ArrayRef<enzyme::Activity> argumentActivity)
      : DenseForwardDataFlowAnalysis(solver), entryBlock(entryBlock),
        argumentActivity(argumentActivity) {}

  void visitOperation(Operation *op, const ForwardMemoryActivity &before,
                      ForwardMemoryActivity *after) override {
    ChangeResult result = after->join(before);
    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    // If we can't reason about the memory effects, then conservatively assume
    // we can't deduce anything about activity via side-effects.
    if (!memory)
      return;

    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);

    for (const auto &effect : effects) {
      Value value = effect.getValue();

      // If we see an effect on anything other than a value, assume we can't
      // deduce anything about the activity.
      if (!value)
        return;

      // In forward-flow, a value is active if loaded from a memory resource
      // that has previously been actively stored to.
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
        forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
          if (before.hasActiveData(alloc)) {
            for (OpResult opResult : op->getResults()) {
              // Mark the result as (forward) active
              // TODO: We might need type analysis here
              // Structs and tensors also have value semantics
              if (isa<FloatType, ComplexType>(opResult.getType())) {
                auto *valueState = getOrCreate<ForwardValueActivity>(opResult);
                propagateIfChanged(
                    valueState,
                    valueState->join(ValueActivity::getActiveVal()));
              }
            }

            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
              // propagate from input to block argument
              for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
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
          if (valueState->getValue().isActiveVal()) {
            auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
            forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
              // Mark the pointer as having been actively stored into
              result |= after->setActiveIn(alloc);
            });
          }
        } else if (auto copySource = getCopySource(op)) {
          auto *srcAliasClass =
              getOrCreateFor<AliasClassLattice>(op, *copySource);
          forEachAliasedAlloc(srcAliasClass, [&](DistinctAttr srcAlloc) {
            if (before.hasActiveData(srcAlloc)) {
              auto *destAliasClass =
                  getOrCreateFor<AliasClassLattice>(op, value);
              forEachAliasedAlloc(destAliasClass, [&](DistinctAttr destAlloc) {
                result |= after->setActiveIn(destAlloc);
              });
            }
          });
        } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
          // linalg.yield stores to the corresponding value.
          for (OpOperand &dpsInit : linalgOp.getDpsInitsMutable()) {
            if (dpsInit.get() == value) {
              int64_t resultIndex =
                  dpsInit.getOperandNumber() - linalgOp.getNumDpsInputs();
              Value yieldOperand =
                  linalgOp.getBlock()->getTerminator()->getOperand(resultIndex);
              auto *valueState =
                  getOrCreateFor<ForwardValueActivity>(op, yieldOperand);
              if (valueState->getValue().isActiveVal()) {
                auto *ptrAliasClass =
                    getOrCreateFor<AliasClassLattice>(op, value);
                forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
                  result |= after->setActiveIn(alloc);
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

  /// Initialize the entry block with the supplied argument activities.
  void setToEntryState(ForwardMemoryActivity *lattice) override {
    if (auto *block = dyn_cast_if_present<Block *>(lattice->getPoint())) {
      if (block == entryBlock) {
        for (const auto &[arg, activity] :
             llvm::zip(block->getArguments(), argumentActivity)) {
          if (activity == enzyme::Activity::enzyme_dup ||
              activity == enzyme::Activity::enzyme_dupnoneed) {
            auto *argAliasClasses =
                getOrCreateFor<AliasClassLattice>(block, arg);
            for (DistinctAttr argAliasClass :
                 argAliasClasses->getAliasClasses()) {
              propagateIfChanged(lattice, lattice->setActiveIn(argAliasClass));
            }
          }
        }
      }
    }
  }

private:
  // A pointer to the entry block and argument activities of the top-level
  // function being differentiated. This is used to set the entry state because
  // we need access to the results of points-to analysis.
  Block *entryBlock;
  ArrayRef<enzyme::Activity> argumentActivity;
};

class DenseBackwardActivityAnalysis
    : public DenseBackwardDataFlowAnalysis<BackwardMemoryActivity> {
public:
  DenseBackwardActivityAnalysis(DataFlowSolver &solver,
                                SymbolTableCollection &symbolTable,
                                FunctionOpInterface parentOp,
                                ArrayRef<enzyme::Activity> argumentActivity)
      : DenseBackwardDataFlowAnalysis(solver, symbolTable), parentOp(parentOp),
        argumentActivity(argumentActivity) {}

  void visitOperation(Operation *op, const BackwardMemoryActivity &after,
                      BackwardMemoryActivity *before) override {
    // Initialize the return activity of arguments.
    if (op->hasTrait<OpTrait::ReturnLike>() && op->getParentOp() == parentOp) {
      for (const auto &[arg, argActivity] : llvm::zip(
               parentOp->getRegions().front().getArguments(), argumentActivity))
        if (argActivity == enzyme::Activity::enzyme_dup ||
            argActivity == enzyme::Activity::enzyme_dupnoneed) {
          auto *argAliasClasses = getOrCreateFor<AliasClassLattice>(op, arg);
          for (DistinctAttr argAliasClass :
               argAliasClasses->getAliasClasses()) {
            propagateIfChanged(before, before->setActiveOut(argAliasClass));
          }
        }

      // Initialize the return activity of the operands
      for (Value operand : op->getOperands()) {
        if (isa<MemRefType, LLVM::LLVMPointerType>(operand.getType())) {
          auto *retAliasClasses =
              getOrCreateFor<AliasClassLattice>(op, operand);
          for (DistinctAttr retAliasClass : retAliasClasses->getAliasClasses())
            propagateIfChanged(before, before->setActiveOut(retAliasClass));
        }
      }
    }

    ChangeResult result = before->meet(after);
    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    // If we can't reason about the memory effects, then conservatively assume
    // we can't deduce anything about activity via side-effects.
    if (!memory)
      return;

    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);

    for (const auto &effect : effects) {
      Value value = effect.getValue();

      // If we see an effect on anything other than a value, assume we can't
      // deduce anything about the activity.
      if (!value)
        return;

      // In backward-flow, a value is active if stored into a memory resource
      // that has subsequently been actively loaded from.
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        for (Value opResult : op->getResults()) {
          auto *valueState =
              getOrCreateFor<BackwardValueActivity>(op, opResult);
          if (valueState->getValue().isActiveVal()) {
            auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
            forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
              result |= before->setActiveOut(alloc);
            });
          }
        }
      }
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(op, value);
        std::optional<Value> stored = getStored(op);
        std::optional<Value> copySource = getCopySource(op);
        forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
          if (stored.has_value() && after.activeDataFlowsOut(alloc)) {
            if (isa<FloatType, ComplexType>(stored->getType())) {
              auto *valueState = getOrCreate<BackwardValueActivity>(*stored);
              propagateIfChanged(
                  valueState, valueState->meet(ValueActivity::getActiveVal()));
            }
          } else if (copySource.has_value() &&
                     after.activeDataFlowsOut(alloc)) {
            auto *srcAliasClass =
                getOrCreateFor<AliasClassLattice>(op, *copySource);
            forEachAliasedAlloc(srcAliasClass, [&](DistinctAttr srcAlloc) {
              before->setActiveOut(srcAlloc);
            });
          } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            if (after.activeDataFlowsOut(alloc)) {
              for (OpOperand &dpsInit : linalgOp.getDpsInitsMutable()) {
                if (dpsInit.get() == value) {
                  int64_t resultIndex =
                      dpsInit.getOperandNumber() - linalgOp.getNumDpsInputs();
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
  FunctionOpInterface parentOp;
  ArrayRef<enzyme::Activity> argumentActivity;
};

void traverseCallGraph(FunctionOpInterface root,
                       SymbolTableCollection *symbolTable,
                       function_ref<void(FunctionOpInterface)> processFunc) {
  std::deque<FunctionOpInterface> frontier{root};
  DenseSet<FunctionOpInterface> visited{root};

  while (!frontier.empty()) {
    FunctionOpInterface curr = frontier.front();
    frontier.pop_front();
    processFunc(curr);

    curr.walk([&](CallOpInterface call) {
      auto neighbor = dyn_cast_if_present<FunctionOpInterface>(
          call.resolveCallable(symbolTable));
      if (neighbor && !visited.contains(neighbor)) {
        frontier.push_back(neighbor);
        visited.insert(neighbor);
      }
    });
  }
}

void printActivityAnalysisResults(const DataFlowSolver &solver,
                                  FunctionOpInterface callee,
                                  const SmallPtrSet<Operation *, 2> &returnOps,
                                  SymbolTableCollection *symbolTable,
                                  bool verbose, bool annotate) {
  auto isActiveData = [&](Value value) {
    auto fva = solver.lookupState<ForwardValueActivity>(value);
    auto bva = solver.lookupState<BackwardValueActivity>(value);
    bool forwardActive = fva && fva->getValue().isActiveVal();
    bool backwardActive = bva && bva->getValue().isActiveVal();
    return forwardActive && backwardActive;
  };

  auto isConstantValue = [&](Value value) {
    // TODO: integers/vectors that might be pointers
    if (isa<LLVM::LLVMPointerType, MemRefType>(value.getType())) {
      assert(returnOps.size() == 1);
      auto *fma = solver.lookupState<ForwardMemoryActivity>(*returnOps.begin());
      auto *bma = solver.lookupState<BackwardMemoryActivity>(
          &callee.getFunctionBody().front().front());

      auto *pointsToSets =
          solver.lookupState<enzyme::PointsToSets>(*returnOps.begin());
      auto *aliasClassLattice = solver.lookupState<AliasClassLattice>(value);
      // Traverse the points-to sets in a simple BFS
      std::deque<DistinctAttr> frontier;
      DenseSet<DistinctAttr> visited;
      const DenseSet<DistinctAttr> &aliasClasses =
          aliasClassLattice->getAliasClasses();
      frontier.insert(frontier.end(), aliasClasses.begin(), aliasClasses.end());
      visited.insert(aliasClasses.begin(), aliasClasses.end());
      while (!frontier.empty()) {
        DistinctAttr aliasClass = frontier.front();
        frontier.pop_front();

        // It's an active pointer if active data flows in from the forward
        // direction and out from the backward direction.
        if (fma->hasActiveData(aliasClass) &&
            bma->activeDataFlowsOut(aliasClass))
          return false;

        // Or if it points to a pointer that points to active data
        for (DistinctAttr neighbor :
             pointsToSets->getPointsTo(aliasClass).getAliasClasses()) {
          if (!visited.contains(neighbor)) {
            visited.insert(neighbor);
            frontier.push_back(neighbor);
          }
        }
      }
      // Otherwise, it's constant
      return true;
    }

    return !isActiveData(value);
  };

  auto isConstantInstruction = [&](Operation *op) {
    return llvm::none_of(op->getOperands(), isActiveData) &&
           llvm::none_of(op->getResults(), isActiveData);
  };

  errs() << FlatSymbolRefAttr::get(callee) << ":\n";
  for (BlockArgument arg : callee.getArguments()) {
    if (Attribute tagAttr =
            callee.getArgAttr(arg.getArgNumber(), "enzyme.tag")) {
      errs() << "  " << tagAttr << ": "
             << (isConstantValue(arg) ? "Constant" : "Active") << "\n";
    }
  }

  if (annotate) {
    MLIRContext *ctx = callee.getContext();
    traverseCallGraph(callee, symbolTable, [&](FunctionOpInterface func) {
      func.walk([&](Operation *op) {
        if (op == func) {
          SmallVector<bool> argICVs(func.getNumArguments());
          llvm::transform(func.getArguments(), argICVs.begin(),
                          isConstantValue);
          func->setAttr("enzyme.icv", DenseBoolArrayAttr::get(ctx, argICVs));
          return;
        }

        op->setAttr("enzyme.ici",
                    BoolAttr::get(ctx, isConstantInstruction(op)));

        bool icv;
        if (op->getNumResults() == 0) {
          icv = true;
        } else if (op->getNumResults() == 1) {
          icv = isConstantValue(op->getResult(0));
        } else {
          op->emitWarning(
              "annotating icv for op that produces multiple results");
          icv = false;
        }
        op->setAttr("enzyme.icv", BoolAttr::get(ctx, icv));
      });
    });
  }
  callee.walk([&](Operation *op) {
    if (op->hasAttr("tag")) {
      errs() << "  " << op->getAttr("tag") << ": ";
      for (OpResult opResult : op->getResults()) {
        errs() << (isConstantValue(opResult) ? "Constant" : "Active") << "\n";
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
      auto *state = solver.lookupState<ForwardMemoryActivity>(returnOp);
      if (state)
        errs() << "forward end state:\n" << *state << "\n";
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

void enzyme::runDataFlowActivityAnalysis(
    FunctionOpInterface callee, ArrayRef<enzyme::Activity> argumentActivity,
    bool print, bool verbose, bool annotate) {
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;

  solver.load<enzyme::PointsToPointerAnalysis>();
  solver.load<enzyme::AliasAnalysis>(callee.getContext());
  solver.load<SparseForwardActivityAnalysis>();
  solver.load<DenseForwardActivityAnalysis>(&callee.getFunctionBody().front(),
                                            argumentActivity);
  solver.load<SparseBackwardActivityAnalysis>(symbolTable);
  solver.load<DenseBackwardActivityAnalysis>(symbolTable, callee,
                                             argumentActivity);

  // Required for the dataflow framework to traverse region-based control flow
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();

  // Initialize the argument states based on the given activity annotations.
  for (const auto &[arg, activity] :
       llvm::zip(callee.getArguments(), argumentActivity)) {
    // enzyme_dup, dupnoneed are initialized within the dense forward/backward
    // analyses, enzyme_const is the default.
    if (activity == enzyme::Activity::enzyme_out) {
      auto *argLattice = solver.getOrCreateState<ForwardValueActivity>(arg);
      argLattice->join(ValueActivity::getActiveVal());
    }
  }

  // Detect function returns as direct children of the FunctionOpInterface
  // that have the ReturnLike trait.
  SmallPtrSet<Operation *, 2> returnOps;
  for (Operation &op : callee.getFunctionBody().getOps()) {
    if (op.hasTrait<OpTrait::ReturnLike>()) {
      returnOps.insert(&op);
      for (Value operand : op.getOperands()) {
        auto *returnLattice =
            solver.getOrCreateState<BackwardValueActivity>(operand);
        // Very basic type inference of the type
        if (isa<FloatType, ComplexType>(operand.getType())) {
          returnLattice->meet(ValueActivity::getActiveVal());
        }
      }
    }
  }

  if (failed(solver.initializeAndRun(callee->getParentOfType<ModuleOp>()))) {
    assert(false && "dataflow analysis failed\n");
  }

  if (print) {
    printActivityAnalysisResults(solver, callee, returnOps, &symbolTable,
                                 verbose, annotate);
  }
}
