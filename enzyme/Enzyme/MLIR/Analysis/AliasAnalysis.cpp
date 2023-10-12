#include "AliasAnalysis.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// The LLVM dialect is only used for the noalias attribute
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// AliasClassLattice
//===----------------------------------------------------------------------===//

void enzyme::AliasClassLattice::print(raw_ostream &os) const {
  os << "size: " << aliasClasses.size() << ":\n";
  for (auto aliasClass : aliasClasses) {
    os << "  " << aliasClass << "\n";
  }
}

AliasResult
enzyme::AliasClassLattice::alias(const AbstractSparseLattice &other) const {
  const auto *rhs = reinterpret_cast<const AliasClassLattice *>(&other);
  if (getPoint() == rhs->getPoint())
    return AliasResult::MustAlias;

  if (isEntry && rhs->isEntry)
    return AliasResult::MayAlias;
  if (isUnknown || rhs->isUnknown)
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

std::optional<Value> enzyme::AliasClassLattice::getCanonicalAllocation() const {
  if (canonicalAllocations.size() == 1) {
    return *canonicalAllocations.begin();
  }
  return std::nullopt;
}

void enzyme::AliasClassLattice::getCanonicalAllocations(
    SmallVectorImpl<Value> &allocations) const {
  allocations.append(canonicalAllocations.begin(), canonicalAllocations.end());
}

ChangeResult
enzyme::AliasClassLattice::join(const AbstractSparseLattice &other) {
  // Set union of the alias classes
  const auto *otherAliasClass =
      reinterpret_cast<const AliasClassLattice *>(&other);
  if (isUnknown) {
    return ChangeResult::NoChange;
  }
  if (otherAliasClass->isUnknown) {
    isUnknown = true;
    return ChangeResult::Change;
  }

  size_t oldSize = aliasClasses.size();
  aliasClasses.insert(otherAliasClass->aliasClasses.begin(),
                      otherAliasClass->aliasClasses.end());

  size_t oldAllocSize = canonicalAllocations.size();
  canonicalAllocations.insert(otherAliasClass->canonicalAllocations.begin(),
                              otherAliasClass->canonicalAllocations.end());

  bool entryChanged = !isEntry && otherAliasClass->isEntry;
  isEntry |= otherAliasClass->isEntry;

  return (oldSize == aliasClasses.size() &&
          oldAllocSize == canonicalAllocations.size() && !entryChanged)
             ? ChangeResult::NoChange
             : ChangeResult::Change;
}

ChangeResult enzyme::AliasClassLattice::markFresh() {
  reset();

  Value value = getPoint();
  auto freshClass = DistinctAttr::create(UnitAttr::get(value.getContext()));
  aliasClasses.insert(freshClass);
  canonicalAllocations.insert(value);
  return ChangeResult::Change;
}

ChangeResult enzyme::AliasClassLattice::markUnknown() {
  if (isUnknown)
    return ChangeResult::NoChange;

  isUnknown = true;
  aliasClasses.clear();
  return ChangeResult::Change;
}

ChangeResult enzyme::AliasClassLattice::markEntry() {
  if (isEntry)
    return ChangeResult::NoChange;

  isEntry = true;
  aliasClasses.clear();
  canonicalAllocations.clear();
  return ChangeResult::Change;
}

ChangeResult enzyme::AliasClassLattice::reset() {
  if (aliasClasses.empty() && canonicalAllocations.empty() && !isUnknown &&
      !isEntry) {
    return ChangeResult::NoChange;
  }
  isUnknown = false;
  isEntry = false;
  aliasClasses.clear();
  canonicalAllocations.clear();
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
        propagateIfChanged(lattice, lattice->markFresh());
      } else {
        propagateIfChanged(lattice, lattice->markEntry());
      }
    }
  } else {
    propagateIfChanged(lattice, lattice->reset());
  }
}

void enzyme::AliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const AliasClassLattice *> operands,
    ArrayRef<AliasClassLattice *> results) {
  bool readsMemory = false;
  if (auto memory = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);
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
            propagateIfChanged(result, result->markFresh());
          }
        }
      } else if (isa<MemoryEffects::Read>(effect.getEffect())) {
        // If the op reads memory, the results don't necessarily alias with the
        // operands.
        readsMemory = true;
        // Conservatively mark the read results as unknown.
        for (AliasClassLattice *result : results) {
          propagateIfChanged(result, result->markUnknown());
        }
      }
    }
  }

  if (readsMemory)
    return;

  // Conservatively assume all results alias all operands
  for (auto *resultLattice : results) {
    for (const auto *operandLattice : operands) {
      join(resultLattice, *operandLattice);
    }
  }
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
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    if (!value)
      return;
    if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
      // Mark the result of the allocation as a fresh memory location
      for (AliasClassLattice *result : results) {
        if (result->getPoint() == value) {
          propagateIfChanged(result, result->markFresh());
        }
      }
    }
  }

  for (auto *resultLattice : results) {
    for (const auto *operandLattice : operands) {
      join(resultLattice, *operandLattice);
    }
  }
}
