#include "AliasAnalysis.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// The LLVM dialect is only used for the noalias attribute
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// PointsToAnalysis
//===----------------------------------------------------------------------===//

void enzyme::PointsToSets::print(raw_ostream &os) const {
  for (const auto &[srcClass, destClasses] : pointsTo) {
    os << "  " << srcClass << " points to {";
    llvm::interleaveComma(destClasses, os);
    os << "}\n";
  }
}

/// Union for every variable.
ChangeResult enzyme::PointsToSets::join(const AbstractDenseLattice &lattice) {
  const auto &rhs = static_cast<const PointsToSets &>(lattice);
  ChangeResult result = ChangeResult::NoChange;
  for (const auto &[alloc, allocSet] : rhs.pointsTo) {
    auto &lhsSet = pointsTo[alloc];
    size_t oldSize = lhsSet.size();
    lhsSet.insert(allocSet.begin(), allocSet.end());
    result |= (lhsSet.size() == oldSize) ? ChangeResult::NoChange
                                         : ChangeResult::Change;
  }
  return result;
}

/// Mark the pointer stored in `dest` as possibly pointing to any of `values`.
ChangeResult
enzyme::PointsToSets::insert(DistinctAttr dest,
                             const DenseSet<DistinctAttr> &values) {
  auto &destPointsTo = pointsTo[dest];
  size_t oldSize = destPointsTo.size();
  destPointsTo.insert(values.begin(), values.end());
  return oldSize == destPointsTo.size() ? ChangeResult::NoChange
                                        : ChangeResult::Change;
}

// TODO: Reduce code duplication with activity analysis
std::optional<Value> getStored(Operation *op);

void enzyme::PointsToPointerAnalysis::visitOperation(Operation *op,
                                                     const PointsToSets &before,
                                                     PointsToSets *after) {
  using llvm::errs;
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

    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      if (auto stored = getStored(op)) {
        auto *srcClasses = getOrCreateFor<AliasClassLattice>(op, *stored);
        auto *destClasses = getOrCreateFor<AliasClassLattice>(op, value);
        if (srcClasses->isUnknown()) {
          errs() << "unimplemented unknown\n";
        }
        if (destClasses->isUnknown()) {
          errs() << "unimplemented unknown\n";
        }
        for (DistinctAttr destClass : destClasses->aliasClasses) {
          propagateIfChanged(
              after, after->insert(destClass, srcClasses->aliasClasses));
        }
      }
    }
  }
}

void enzyme::PointsToPointerAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, CallControlFlowAction action,
    const PointsToSets &before, PointsToSets *after) {
  join(after, before);
  if (action == CallControlFlowAction::ExternalCallee) {
    auto symbol = cast<SymbolRefAttr>(call.getCallableForCallee());
    if (symbol.getLeafReference().getValue() == "posix_memalign") {
      // memalign deals with nested pointers and thus must be handled here
      // memalign points to a value
      OperandRange arguments = call.getArgOperands();
      auto *memPtr = getOrCreateFor<AliasClassLattice>(call, arguments[0]);
      for (DistinctAttr memPtrClass : memPtr->aliasClasses) {
        auto debugLabel = StringAttr::get(call.getContext(), "memalign");
        propagateIfChanged(
            after, after->insert(memPtrClass,
                                 {AliasClassLattice::getFresh(debugLabel)}));
      }
    }
  }
}

// The default initialization (empty map of empty sets) is correct.
void enzyme::PointsToPointerAnalysis::setToEntryState(PointsToSets *lattice) {}

//===----------------------------------------------------------------------===//
// AliasClassLattice
//===----------------------------------------------------------------------===//

void enzyme::AliasClassLattice::print(raw_ostream &os) const {
  if (unknown) {
    os << "Unknown AC";
  } else {
    os << "size: " << aliasClasses.size() << ":\n";
    for (auto aliasClass : aliasClasses) {
      os << "  " << aliasClass << "\n";
    }
  }
}

AliasResult
enzyme::AliasClassLattice::alias(const AbstractSparseLattice &other) const {
  const auto *rhs = reinterpret_cast<const AliasClassLattice *>(&other);
  if (getPoint() == rhs->getPoint())
    return AliasResult::MustAlias;

  if (unknown || rhs->unknown)
    return AliasResult::MayAlias;

  size_t overlap = llvm::count_if(aliasClasses, [rhs](DistinctAttr aliasClass) {
    return rhs->aliasClasses.contains(aliasClass);
  });

  if (overlap == 0)
    return AliasResult::NoAlias;

  // Due to the conservative nature of propagation from all operands to all
  // results, we can't actually assume that exactly identical alias classes will
  // lead to a "must alias" result.

  // if (overlap == aliasClasses.size() &&
  //  overlap == rhs->aliasClasses.size())
  //    return AliasResult::MustAlias;

  return AliasResult::MayAlias;
}

ChangeResult
enzyme::AliasClassLattice::join(const AbstractSparseLattice &other) {
  // Set union of the alias classes
  const auto *otherAliasClass =
      reinterpret_cast<const AliasClassLattice *>(&other);
  if (unknown) {
    return ChangeResult::NoChange;
  }
  if (otherAliasClass->unknown) {
    unknown = true;
    return ChangeResult::Change;
  }

  return insert(otherAliasClass->aliasClasses);
}

ChangeResult
enzyme::AliasClassLattice::insert(const DenseSet<DistinctAttr> &classes) {
  size_t oldSize = aliasClasses.size();
  aliasClasses.insert(classes.begin(), classes.end());
  return aliasClasses.size() == oldSize ? ChangeResult::NoChange
                                        : ChangeResult::Change;
}

ChangeResult enzyme::AliasClassLattice::markFresh(Attribute debugLabel) {
  reset();

  Value value = getPoint();
  if (!debugLabel)
    debugLabel = UnitAttr::get(value.getContext());
  auto freshClass = AliasClassLattice::getFresh(debugLabel);
  aliasClasses.insert(freshClass);
  return ChangeResult::Change;
}

ChangeResult enzyme::AliasClassLattice::markUnknown() {
  if (unknown)
    return ChangeResult::NoChange;

  unknown = true;
  aliasClasses.clear();
  return ChangeResult::Change;
}

ChangeResult enzyme::AliasClassLattice::reset() {
  if (aliasClasses.empty() && !unknown) {
    return ChangeResult::NoChange;
  }
  unknown = false;
  aliasClasses.clear();
  return ChangeResult::Change;
}

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

void enzyme::AliasAnalysis::setToEntryState(AliasClassLattice *lattice) {
  if (auto arg = dyn_cast<BlockArgument>(lattice->getPoint())) {
    if (auto funcOp =
            dyn_cast<FunctionOpInterface>(arg.getOwner()->getParentOp())) {
      if (funcOp.getArgAttr(arg.getArgNumber(),
                            LLVM::LLVMDialect::getNoAliasAttrName())) {
        Attribute debugLabel =
            funcOp.getArgAttr(arg.getArgNumber(), "enzyme.tag");
        propagateIfChanged(lattice, lattice->markFresh(debugLabel));
      } else {
        // TODO: Not safe in general, integers can be a result of ptrtoint. We
        // need a type analysis here I guess?
        if (isa<LLVM::LLVMPointerType, MemRefType>(arg.getType()))
          propagateIfChanged(lattice, lattice->insert({entryClass}));
      }
    }
  } else {
    propagateIfChanged(lattice, lattice->reset());
  }
}

void enzyme::AliasAnalysis::transfer(
    Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects,
    ArrayRef<const AliasClassLattice *> operands,
    ArrayRef<AliasClassLattice *> results) {
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    if (!value) {
      // TODO: we can't assume anything about entry states
      continue;
    }

    if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
      // Mark the result of the allocation as a fresh memory location
      for (AliasClassLattice *result : results) {
        if (result->getPoint() == value) {
          Attribute debugLabel = op->getAttr("tag");
          propagateIfChanged(result, result->markFresh(debugLabel));
        }
      }
    } else if (isa<MemoryEffects::Read>(effect.getEffect())) {
      auto *pointsToSets = getOrCreateFor<PointsToSets>(op, op);
      for (auto srcClass : getLatticeElement(value)->aliasClasses) {
        const auto &srcPointsTo = pointsToSets->pointsTo.lookup(srcClass);
        for (AliasClassLattice *result : results) {
          propagateIfChanged(result, result->insert(srcPointsTo));
        }
      }
    }
  }

  if (!effects.empty())
    return;

  // For operations that don't touch memory, conservatively assume all results
  // alias all operands
  for (auto *resultLattice : results) {
    for (const auto *operandLattice : operands) {
      join(resultLattice, *operandLattice);
    }
  }
}

void enzyme::AliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const AliasClassLattice *> operands,
    ArrayRef<AliasClassLattice *> results) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  if (auto memory = dyn_cast<MemoryEffectOpInterface>(op))
    memory.getEffects(effects);

  transfer(op, effects, operands, results);
  if (auto view = dyn_cast<OffsetSizeAndStrideOpInterface>(op)) {
    // TODO: special handling for offset size and stride op interface to prove
    // that non-overlapping subviews of the same buffer don't alias could be a
    // promising extension.
  }
}

// TODO: Move this somewhere shared
void getEffectsForExternalCall(
    CallOpInterface call,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee())) {
    StringRef callableName = symbol.getLeafReference().getValue();
    if (callableName == "malloc" || callableName == "_Znwm") {
      assert(call->getNumResults() == 1);
      effects.push_back(MemoryEffects::EffectInstance(
          MemoryEffects::Allocate::get(), call->getResult(0)));
    }
  }
}

void enzyme::AliasAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const AliasClassLattice *> operands,
    ArrayRef<AliasClassLattice *> results) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  getEffectsForExternalCall(call, effects);
  transfer(call, effects, operands, results);
}
