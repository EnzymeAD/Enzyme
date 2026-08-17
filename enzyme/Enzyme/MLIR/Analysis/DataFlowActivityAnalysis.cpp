//===- DataFlowActivityAnalysis.h - Implementation of Activity Analysis ---===//
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
// This file contains the implementation of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file formulates activity analysis within
// a dataflow framework.
//
//===----------------------------------------------------------------------===//
#include "DataFlowActivityAnalysis.h"
#include "DataFlowAliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

// TODO: Don't depend on specific dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"

using namespace mlir;
using namespace mlir::dataflow;
using enzyme::AliasClassLattice;

raw_ostream &operator<<(raw_ostream &os, const enzyme::ValueActivity &val) {
  val.print(os);
  return os;
}

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

//===----------------------------------------------------------------------===//
// ValueActivity
//===----------------------------------------------------------------------===//
raw_ostream &enzyme::ValueActivity::operator<<(raw_ostream &os) const {
  print(os);
  return os;
}

void enzyme::ValueActivity::print(llvm::raw_ostream &os) const {
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

//===----------------------------------------------------------------------===//
// BackwardValueActivity
//===----------------------------------------------------------------------===//
ChangeResult
enzyme::BackwardValueActivity::meet(const AbstractSparseLattice &other) {
  const auto *rhs =
      reinterpret_cast<const enzyme::BackwardValueActivity *>(&other);
  return meet(rhs->getValue());
}

ChangeResult enzyme::BackwardValueActivity::meet(enzyme::ValueActivity other) {
  auto met = enzyme::ValueActivity::merge(getValue(), other);
  if (getValue() == met) {
    return ChangeResult::NoChange;
  }

  value = met;
  return ChangeResult::Change;
}

void enzyme::BackwardValueActivity::print(raw_ostream &os) const {
  value.print(os);
}

//===----------------------------------------------------------------------===//
// SparseForwardActivityAnalysis
//===----------------------------------------------------------------------===//

/// In general, we don't know anything about entry operands.
void enzyme::SparseForwardActivityAnalysis::setToEntryState(
    enzyme::ForwardValueActivity *lattice) {
  // llvm::errs() << "sparse forward setting to entry state\n";
  propagateIfChanged(lattice, lattice->join(enzyme::ValueActivity()));
}

LogicalResult enzyme::SparseForwardActivityAnalysis::visitOperation(
    Operation *op, ArrayRef<const enzyme::ForwardValueActivity *> operands,
    ArrayRef<enzyme::ForwardValueActivity *> results) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return success();

  // Bail out if this op affects memory.
  if (!isPure(op))
    return success();

  transfer(op, operands, results);

  return success();
}

void enzyme::SparseForwardActivityAnalysis::visitExternalCall(
    CallOpInterface call,
    ArrayRef<const enzyme::ForwardValueActivity *> operands,
    ArrayRef<enzyme::ForwardValueActivity *> results) {
  transfer(call, operands, results);
}

void enzyme::SparseForwardActivityAnalysis::transfer(
    Operation *op, ArrayRef<const enzyme::ForwardValueActivity *> operands,
    ArrayRef<enzyme::ForwardValueActivity *> results) {
  // For value-based AA, assume any active argument leads to an active
  // result.
  enzyme::ValueActivity joinedResult;
  for (const enzyme::ForwardValueActivity *operand : operands)
    joinedResult =
        enzyme::ValueActivity::merge(joinedResult, operand->getValue());

  // Only mark results as active data if the type can carry perturbations and
  // has value semantics
  for (enzyme::ForwardValueActivity *result : results) {
    if (joinedResult.isActiveVal())
      propagateIfChanged(
          result, result->join(
                      isa<FloatType, ComplexType>(result->getAnchor().getType())
                          ? joinedResult
                          : enzyme::ValueActivity::getConstant()));
    else
      propagateIfChanged(result, result->join(joinedResult));
  }
}

//===----------------------------------------------------------------------===//
// SparseBackwardActivityAnalysis
//===----------------------------------------------------------------------===//

void enzyme::SparseBackwardActivityAnalysis::transfer(
    Operation *op, ArrayRef<enzyme::BackwardValueActivity *> operands,
    ArrayRef<const enzyme::BackwardValueActivity *> results) {
  // Propagate all operands to all results
  for (auto operand : operands)
    for (auto result : results)
      meet(operand, *result);
}

LogicalResult enzyme::SparseBackwardActivityAnalysis::visitOperation(
    Operation *op, ArrayRef<enzyme::BackwardValueActivity *> operands,
    ArrayRef<const enzyme::BackwardValueActivity *> results) {
  // Bail out if the op propagates memory
  if (!isPure(op)) {
    return success();
  }

  transfer(op, operands, results);
  return success();
}

void enzyme::SparseBackwardActivityAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<enzyme::BackwardValueActivity *> operands,
    ArrayRef<const enzyme::BackwardValueActivity *> results) {
  transfer(call, operands, results);
}

// When applying a transfer function to a store from memory, we need to know
// what value is being stored.
std::optional<Value> getStored(Operation *op) {
  if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
    return storeOp.getValue();
  } else if (auto memsetOp = dyn_cast<LLVM::MemsetOp>(op)) {
    return memsetOp.getVal();
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return storeOp.getValue();
  } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
    return storeOp.getValue();
  } else if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
    return pushOp.getValue();
  }
  return std::nullopt;
}

// TODO consider making this an interface ourselves
std::optional<Value> getCopySource(Operation *op) {
  if (isa<LLVM::MemcpyOp, LLVM::MemcpyInlineOp, LLVM::MemmoveOp>(op)) {
    return op->getOperand(1);
  }
  return std::nullopt;
}

/// The dense analyses operate using a pointer's "canonical allocation", the
/// Value corresponding to its allocation.
/// The callback may receive null allocation when the class alias set is
/// unknown.
/// If the classes are undefined, the callback will not be called at all.
void forEachAliasedAlloc(const AliasClassLattice *ptrAliasClass,
                         function_ref<void(DistinctAttr)> forEachFn) {
  (void)ptrAliasClass->getAliasClassesObject().foreachElement(
      [&](DistinctAttr alloc, enzyme::AliasClassSet::State state) {
        if (state != enzyme::AliasClassSet::State::Undefined)
          forEachFn(alloc);
        return ChangeResult::NoChange;
      });
}

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
          call.resolveCallableInTable(symbolTable));
      if (neighbor && !visited.contains(neighbor)) {
        frontier.push_back(neighbor);
        visited.insert(neighbor);
      }
    });
  }
}

//===----------------------------------------------------------------------===//
// MemoryActivityState
//===----------------------------------------------------------------------===//
ChangeResult enzyme::MemoryActivityState::reset() {
  if (!activeIn && !activeOut)
    return ChangeResult::NoChange;
  activeIn = false;
  activeOut = false;
  return ChangeResult::Change;
}

ChangeResult
enzyme::MemoryActivityState::merge(const enzyme::MemoryActivityState &other) {
  if (*this == other) {
    return ChangeResult::NoChange;
  }

  activeIn |= other.activeIn;
  activeOut |= other.activeOut;
  return ChangeResult::Change;
}

//===----------------------------------------------------------------------===//
// MemoryActivity
//===----------------------------------------------------------------------===//

ChangeResult enzyme::MemoryActivity::reset() {
  if (activityStates.empty())
    return otherMemoryActivity.reset();
  activityStates.clear();
  return otherMemoryActivity.reset();
}

bool enzyme::MemoryActivity::hasActiveData(DistinctAttr aliasClass) const {
  if (!aliasClass)
    return otherMemoryActivity.activeIn;
  auto it = activityStates.find(aliasClass);
  if (it != activityStates.end())
    return it->getSecond().activeIn;
  return otherMemoryActivity.activeIn;
}

bool enzyme::MemoryActivity::activeDataFlowsOut(DistinctAttr aliasClass) const {
  if (!aliasClass)
    return otherMemoryActivity.activeOut;

  auto it = activityStates.find(aliasClass);
  if (it != activityStates.end())
    return it->getSecond().activeOut;
  return otherMemoryActivity.activeOut;
}

/// Set the internal activity state. Accepts null attribute to indicate "other
/// classes".
ChangeResult enzyme::MemoryActivity::setActiveIn(DistinctAttr aliasClass) {
  if (!aliasClass)
    return setActiveIn();

  auto &state = activityStates[aliasClass];
  ChangeResult result =
      state.activeIn ? ChangeResult::NoChange : ChangeResult::Change;
  state.activeIn = true;
  return result;
}
ChangeResult enzyme::MemoryActivity::setActiveIn() {
  if (otherMemoryActivity.activeIn && activityStates.empty())
    return ChangeResult::NoChange;
  otherMemoryActivity.activeIn = true;
  activityStates.clear();
  return ChangeResult::Change;
}
ChangeResult enzyme::MemoryActivity::setActiveOut(DistinctAttr aliasClass) {
  if (!aliasClass)
    return setActiveOut();

  auto &state = activityStates[aliasClass];
  ChangeResult result =
      state.activeOut ? ChangeResult::NoChange : ChangeResult::Change;
  state.activeOut = true;
  return result;
}
ChangeResult enzyme::MemoryActivity::setActiveOut() {
  if (otherMemoryActivity.activeOut && activityStates.empty())
    return ChangeResult::NoChange;
  otherMemoryActivity.activeOut = true;
  activityStates.clear();
  return ChangeResult::Change;
}

void enzyme::MemoryActivity::print(raw_ostream &os) const {
  if (activityStates.empty()) {
    os << "<memory activity state was empty>"
       << "\n";
  }
  for (const auto &[value, state] : activityStates) {
    os << value << ": in " << state.activeIn << " out " << state.activeOut
       << "\n";
  }
  os << "other classes: in " << otherMemoryActivity.activeIn << " out "
     << otherMemoryActivity.activeOut << "\n";
}

raw_ostream &enzyme::MemoryActivity::operator<<(raw_ostream &os) const {
  print(os);
  return os;
}

ChangeResult
enzyme::MemoryActivity::merge(const AbstractDenseLattice &lattice) {

  const auto &rhs = static_cast<const MemoryActivity &>(lattice);
  ChangeResult result = ChangeResult::NoChange;
  DenseSet<DistinctAttr> known;
  auto lhsRange = llvm::make_first_range(activityStates);
  auto rhsRange = llvm::make_first_range(rhs.activityStates);
  known.insert(lhsRange.begin(), lhsRange.end());
  known.insert(rhsRange.begin(), rhsRange.end());

  MemoryActivityState updatedOther(otherMemoryActivity);
  result |= updatedOther.merge(rhs.otherMemoryActivity);
  DenseMap<DistinctAttr, MemoryActivityState> updated;
  for (DistinctAttr d : known) {
    auto lhsIt = activityStates.find(d);
    auto rhsIt = rhs.activityStates.find(d);
    bool isKnownInLHS = lhsIt != activityStates.end();
    bool isKnownInRHS = rhsIt != rhs.activityStates.end();
    const MemoryActivityState *lhsActivity =
        isKnownInLHS ? &lhsIt->getSecond() : &otherMemoryActivity;
    const MemoryActivityState *rhsActivity =
        isKnownInRHS ? &rhsIt->getSecond() : &rhs.otherMemoryActivity;
    MemoryActivityState updatedActivity(*lhsActivity);
    (void)updatedActivity.merge(*rhsActivity);
    if ((lhsIt != activityStates.end() &&
         updatedActivity != lhsIt->getSecond()) ||
        (lhsIt == activityStates.end() &&
         updatedActivity != otherMemoryActivity)) {
      result |= ChangeResult::Change;
    }
    if (updatedActivity != updatedOther)
      updated.try_emplace(d, updatedActivity);
  }

  std::swap(updated, activityStates);
  return otherMemoryActivity.merge(rhs.otherMemoryActivity) | result;
}

//===----------------------------------------------------------------------===//
// DenseForwardActivityAnalysis
//===----------------------------------------------------------------------===//

LogicalResult enzyme::DenseForwardActivityAnalysis::visitOperation(
    Operation *op, const ForwardMemoryActivity &before,
    ForwardMemoryActivity *after) {
  join(after, before);
  ChangeResult result = ChangeResult::NoChange;

  // TODO If we know this is inactive by definition
  // if (auto ifaceOp = dyn_cast<enzyme::ActivityOpInterface>(op)) {
  //   if (ifaceOp.isInactive()) {
  //     propagateIfChanged(after, result);
  //     return;
  //   }
  // }

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  // If we can't reason about the memory effects, then conservatively assume
  // we can't deduce anything about activity via side-effects.
  if (!memory)
    return success();

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);

  for (const auto &effect : effects) {
    Value value = effect.getValue();

    // If we see an effect on anything other than a value, assume we can't
    // deduce anything about the activity.
    if (!value)
      return success();

    // In forward-flow, a value is active if loaded from a memory resource
    // that has previously been actively stored to.
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      auto *ptrAliasClass =
          getOrCreateFor<AliasClassLattice>(getProgramPointAfter(op), value);
      forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
        if (before.hasActiveData(alloc)) {
          for (OpResult opResult : op->getResults()) {
            // Mark the result as (forward) active
            // TODO: We might need type analysis here
            // Structs and tensors also have value semantics
            if (isa<FloatType, ComplexType>(opResult.getType())) {
              auto *valueState =
                  getOrCreate<enzyme::ForwardValueActivity>(opResult);
              propagateIfChanged(
                  valueState,
                  valueState->join(enzyme::ValueActivity::getActiveVal()));
            }
          }

          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            // propagate from input to block argument
            for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
              if (inputOperand->get() == value) {
                auto *valueState = getOrCreate<enzyme::ForwardValueActivity>(
                    linalgOp.getMatchingBlockArgument(inputOperand));
                propagateIfChanged(
                    valueState,
                    valueState->join(enzyme::ValueActivity::getActiveVal()));
              }
            }
          }
        }
      });
    }

    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      std::optional<Value> stored = getStored(op);
      if (stored.has_value()) {
        auto *valueState = getOrCreateFor<enzyme::ForwardValueActivity>(
            getProgramPointAfter(op), *stored);
        if (valueState->getValue().isActiveVal()) {
          auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(
              getProgramPointAfter(op), value);
          forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
            // Mark the pointer as having been actively stored into
            result |= after->setActiveIn(alloc);
          });
        }
      } else if (auto copySource = getCopySource(op)) {
        auto *srcAliasClass = getOrCreateFor<AliasClassLattice>(
            getProgramPointAfter(op), *copySource);
        forEachAliasedAlloc(srcAliasClass, [&](DistinctAttr srcAlloc) {
          if (before.hasActiveData(srcAlloc)) {
            auto *destAliasClass = getOrCreateFor<AliasClassLattice>(
                getProgramPointAfter(op), value);
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
            auto *valueState = getOrCreateFor<enzyme::ForwardValueActivity>(
                getProgramPointAfter(op), yieldOperand);
            if (valueState->getValue().isActiveVal()) {
              auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(
                  getProgramPointAfter(op), value);
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
  return success();
}

void enzyme::DenseForwardActivityAnalysis::setToEntryState(
    ForwardMemoryActivity *lattice) {
  if (auto pp = dyn_cast_if_present<ProgramPoint *>(lattice->getAnchor()))
    if (Block *block = pp->getBlock();
        block && block == entryBlock && pp->isBlockStart()) {
      for (const auto &[arg, activity] :
           llvm::zip(block->getArguments(), argumentActivity)) {
        if (activity != enzyme::Activity::enzyme_dup &&
            activity != enzyme::Activity::enzyme_dupnoneed)
          continue;
        auto *argAliasClasses = getOrCreateFor<AliasClassLattice>(
            getProgramPointBefore(block), arg);
        ChangeResult changed =
            argAliasClasses->getAliasClassesObject().foreachElement(
                [lattice](DistinctAttr argAliasClass,
                          enzyme::AliasClassSet::State state) {
                  if (state == enzyme::AliasClassSet::State::Undefined)
                    return ChangeResult::NoChange;
                  return lattice->setActiveIn(argAliasClass);
                });
        propagateIfChanged(lattice, changed);
      }
    }
}

//===----------------------------------------------------------------------===//
// DenseBackwardActivityAnalysis
//===----------------------------------------------------------------------===//

LogicalResult enzyme::DenseBackwardActivityAnalysis::visitOperation(
    Operation *op, const BackwardMemoryActivity &after,
    BackwardMemoryActivity *before) {

  // TODO: If we know this is inactive by definition
  // if (auto ifaceOp = dyn_cast<enzyme::ActivityOpInterface>(op)) {
  //   if (ifaceOp.isInactive()) {
  //     return;
  //   }
  // }

  // Initialize the return activity of arguments.
  if (op->hasTrait<OpTrait::ReturnLike>() && op->getParentOp() == parentOp) {
    for (const auto &[arg, argActivity] : llvm::zip(
             parentOp->getRegions().front().getArguments(), argumentActivity)) {
      if (argActivity != enzyme::Activity::enzyme_dup &&
          argActivity != enzyme::Activity::enzyme_dupnoneed) {
        continue;
      }
      auto *argAliasClasses =
          getOrCreateFor<AliasClassLattice>(getProgramPointBefore(op), arg);
      ChangeResult changed =
          argAliasClasses->getAliasClassesObject().foreachElement(
              [before](DistinctAttr argAliasClass,
                       enzyme::AliasClassSet::State state) {
                if (state == enzyme::AliasClassSet::State::Undefined)
                  return ChangeResult::NoChange;
                return before->setActiveOut(argAliasClass);
              });
      propagateIfChanged(before, changed);
    }

    assert(op->getNumOperands() == returnActivity.size() &&
           "the number of returned arguments must match the number of return "
           "activity entries");

    // Initialize the return activity of the operands
    for (const auto &[operand, operandActivity] :
         llvm::zip(op->getOperands(), returnActivity)) {
      if (operandActivity != enzyme::Activity::enzyme_dup &&
          operandActivity != enzyme::Activity::enzyme_dupnoneed)
        continue;

      auto *retAliasClasses =
          getOrCreateFor<AliasClassLattice>(getProgramPointBefore(op), operand);
      ChangeResult changed =
          retAliasClasses->getAliasClassesObject().foreachElement(
              [before](DistinctAttr retAliasClass,
                       enzyme::AliasClassSet::State state) {
                if (state == enzyme::AliasClassSet::State::Undefined)
                  return ChangeResult::NoChange;
                return before->setActiveOut(retAliasClass);
              });
      propagateIfChanged(before, changed);
    }
  }

  meet(before, after);
  ChangeResult result = ChangeResult::NoChange;
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  // If we can't reason about the memory effects, then conservatively assume
  // we can't deduce anything about activity via side-effects.
  if (!memory)
    return success();

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);

  for (const auto &effect : effects) {
    Value value = effect.getValue();

    // If we see an effect on anything other than a value, assume we can't
    // deduce anything about the activity.
    if (!value)
      return success();

    // In backward-flow, a value is active if stored into a memory resource
    // that has subsequently been actively loaded from.
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      for (Value opResult : op->getResults()) {
        auto *valueState = getOrCreateFor<enzyme::BackwardValueActivity>(
            getProgramPointBefore(op), opResult);
        if (valueState->getValue().isActiveVal()) {
          auto *ptrAliasClass = getOrCreateFor<AliasClassLattice>(
              getProgramPointBefore(op), value);
          forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
            result |= before->setActiveOut(alloc);
          });
        }
      }
    }
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      auto *ptrAliasClass =
          getOrCreateFor<AliasClassLattice>(getProgramPointBefore(op), value);
      std::optional<Value> stored = getStored(op);
      std::optional<Value> copySource = getCopySource(op);
      forEachAliasedAlloc(ptrAliasClass, [&](DistinctAttr alloc) {
        if (stored.has_value() && after.activeDataFlowsOut(alloc)) {
          if (isa<FloatType, ComplexType>(stored->getType())) {
            auto *valueState =
                getOrCreate<enzyme::BackwardValueActivity>(*stored);
            propagateIfChanged(
                valueState,
                valueState->meet(enzyme::ValueActivity::getActiveVal()));
          }
        } else if (copySource.has_value() && after.activeDataFlowsOut(alloc)) {
          auto *srcAliasClass = getOrCreateFor<AliasClassLattice>(
              getProgramPointBefore(op), *copySource);
          forEachAliasedAlloc(srcAliasClass, [&](DistinctAttr srcAlloc) {
            result |= before->setActiveOut(srcAlloc);
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
                    getOrCreate<enzyme::BackwardValueActivity>(yieldOperand);
                propagateIfChanged(
                    valueState,
                    valueState->meet(enzyme::ValueActivity::getActiveVal()));
              }
            }
          }
        }
      });
    }
  }
  propagateIfChanged(before, result);
  return success();
}

void printActivityAnalysisResults(DataFlowSolver &solver,
                                  FunctionOpInterface callee,
                                  const SmallPtrSet<Operation *, 2> &returnOps,
                                  SymbolTableCollection *symbolTable,
                                  bool verbose, bool annotate) {
  auto isActiveData = [&](Value value) {
    auto fva = solver.lookupState<enzyme::ForwardValueActivity>(value);
    auto bva = solver.lookupState<enzyme::BackwardValueActivity>(value);
    bool forwardActive = fva && fva->getValue().isActiveVal();
    bool backwardActive = bva && bva->getValue().isActiveVal();
    return forwardActive && backwardActive;
  };

  auto isConstantValue = [&](Value value) {
    // TODO: integers/vectors that might be pointers
    if (isa<LLVM::LLVMPointerType, MemRefType>(value.getType())) {
      assert(returnOps.size() == 1);
      auto *fma = solver.lookupState<enzyme::ForwardMemoryActivity>(
          solver.getProgramPointAfter(*returnOps.begin()));
      auto *bma = solver.lookupState<enzyme::BackwardMemoryActivity>(
          solver.getProgramPointBefore(
              &callee.getFunctionBody().front().front()));

      const enzyme::PointsToSets *pointsToSets =
          solver.lookupState<enzyme::PointsToSets>(
              solver.getProgramPointAfter(*returnOps.begin()));
      auto *aliasClassLattice = solver.lookupState<AliasClassLattice>(value);
      // Traverse the points-to sets in a simple BFS
      std::deque<DistinctAttr> frontier;
      DenseSet<DistinctAttr> visited;
      auto scheduleVisit = [&](const enzyme::AliasClassSet &aliasClasses) {
        (void)aliasClasses.foreachElement(
            [&](DistinctAttr neighbor, enzyme::AliasClassSet::State state) {
              assert(neighbor &&
                     "unhandled undefined/unknown case before visit");
              if (!visited.contains(neighbor)) {
                visited.insert(neighbor);
                frontier.push_back(neighbor);
              }
              return ChangeResult::NoChange;
            });
      };

      // If this triggers, investigate why the alias classes weren't computed.
      // If they weren't computed legitimately, treat the value as
      // conservatively non-constant or change the return type to be tri-state.
      assert(!aliasClassLattice->isUndefined() &&
             "didn't compute alias classes");

      if (aliasClassLattice->isUnknown()) {
        // Pointers of unknown class may point to active data.
        // TODO: is this overly conservative? Should we rather check
        // if listed classes may point to non-constants?
        return false;
      } else {
        scheduleVisit(aliasClassLattice->getAliasClassesObject());
      }
      while (!frontier.empty()) {
        DistinctAttr aliasClass = frontier.front();
        frontier.pop_front();

        // It's an active pointer if active data flows in from the forward
        // direction and out from the backward direction.
        if (fma->hasActiveData(aliasClass) &&
            bma->activeDataFlowsOut(aliasClass))
          return false;

        // If this triggers, investigate why points-to sets couldn't be
        // computed. Treat conservatively as "unknown" if necessary.
        assert(!pointsToSets->getPointsTo(aliasClass).isUndefined() &&
               "couldn't compute points-to sets");

        // Pointers to unknown classes may (transitively) point to active data.
        if (pointsToSets->getPointsTo(aliasClass).isUnknown())
          return false;

        scheduleVisit(pointsToSets->getPointsTo(aliasClass));
      }
      // Otherwise, it's constant
      return true;
    }

    return !isActiveData(value);
  };

  std::function<bool(Operation *)> isConstantInstruction = [&](Operation *op) {
    if (isPure(op)) {
      // If an operation doesn't have side effects, the only way it can
      // propagate active data is through its results.
      return llvm::none_of(op->getResults(), isActiveData);
    }
    // We need a special case because stores of active pointers don't fit the
    // definition but are active instructions
    if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
      if (!isConstantValue(storeOp.getValue())) {
        return false;
      }
    } else if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      // TODO: Should traverse bottom-up for performance (or cache
      // intermediate results)
      auto callable = cast<CallableOpInterface>(callOp.resolveCallable());
      if (callable.getCallableRegion()) {
        // If any of the instructions in the body are active instructions, the
        // function is active.
        WalkResult result = callable->walk([&](Operation *op) {
          if (!isConstantInstruction(op)) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        return !result.wasInterrupted();
      } else {
        // fall back to seeing if any operand or result is active data
      }
    }
    return llvm::none_of(op->getOperands(), isActiveData) &&
           llvm::none_of(op->getResults(), isActiveData);
  };

  llvm::errs() << FlatSymbolRefAttr::get(callee) << ":\n";
  for (BlockArgument arg : callee.getArguments()) {
    if (Attribute tagAttr =
            callee.getArgAttr(arg.getArgNumber(), "enzyme.tag")) {
      llvm::errs() << "  " << tagAttr << ": "
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
      llvm::errs() << "  " << op->getAttr("tag") << ": ";
      for (OpResult opResult : op->getResults()) {
        llvm::errs() << (isConstantValue(opResult) ? "Constant" : "Active")
                     << "\n";
      }
    }
    if (verbose) {
      // Annotate each op's results with its value activity states
      for (OpResult result : op->getResults()) {
        auto forwardValueActivity =
            solver.lookupState<enzyme::ForwardValueActivity>(result);
        if (forwardValueActivity) {
          std::string dest, key{"fva"};
          llvm::raw_string_ostream os(dest);
          if (op->getNumResults() != 1)
            key += result.getResultNumber();
          forwardValueActivity->getValue().print(os);
          op->setAttr(key, StringAttr::get(op->getContext(), dest));
        }

        auto backwardValueActivity =
            solver.lookupState<enzyme::BackwardValueActivity>(result);
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
          solver.lookupState<enzyme::BackwardValueActivity>(arg);
      if (backwardValueActivity) {
        std::string dest;
        llvm::raw_string_ostream os(dest);
        backwardValueActivity->getValue().print(os);
        callee.setArgAttr(arg.getArgNumber(), "enzyme.bva",
                          StringAttr::get(callee->getContext(), dest));
      }
    }

    for (Operation *returnOp : returnOps) {
      auto *state = solver.lookupState<enzyme::ForwardMemoryActivity>(
          solver.getProgramPointAfter(returnOp));
      if (state)
        llvm::errs() << "forward end state:\n" << *state << "\n";
      else
        llvm::errs() << "state was null\n";
    }

    auto startState = solver.lookupState<enzyme::BackwardMemoryActivity>(
        solver.getProgramPointAfter(&callee.getFunctionBody().front().front()));
    if (startState)
      llvm::errs() << "backwards end state:\n" << *startState << "\n";
    else
      llvm::errs() << "backwards end state was null\n";
  }
}

void enzyme::runDataFlowActivityAnalysis(
    FunctionOpInterface callee, ArrayRef<enzyme::Activity> argumentActivity,
    ArrayRef<enzyme::Activity> returnActivity, bool print, bool verbose,
    bool annotate) {
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;

  solver.load<enzyme::PointsToPointerAnalysis>();
  solver.load<enzyme::AliasAnalysis>(callee.getContext());
  solver.load<SparseForwardActivityAnalysis>();
  solver.load<DenseForwardActivityAnalysis>(&callee.getFunctionBody().front(),
                                            argumentActivity);
  solver.load<SparseBackwardActivityAnalysis>(symbolTable);
  solver.load<DenseBackwardActivityAnalysis>(symbolTable, callee,
                                             argumentActivity, returnActivity);

  // Required for the dataflow framework to traverse region-based control flow
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();

  // Initialize the argument states based on the given activity annotations.
  for (const auto &[arg, activity] :
       llvm::zip(callee.getArguments(), argumentActivity)) {
    // enzyme_dup, dupnoneed are initialized within the dense forward/backward
    // analyses, enzyme_const is the default.
    if (activity == enzyme::Activity::enzyme_active) {
      auto *argLattice =
          solver.getOrCreateState<enzyme::ForwardValueActivity>(arg);
      (void)argLattice->join(ValueActivity::getActiveVal());
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
            solver.getOrCreateState<enzyme::BackwardValueActivity>(operand);
        // Very basic type inference of the type
        if (isa<FloatType, ComplexType>(operand.getType())) {
          (void)returnLattice->meet(ValueActivity::getActiveVal());
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
