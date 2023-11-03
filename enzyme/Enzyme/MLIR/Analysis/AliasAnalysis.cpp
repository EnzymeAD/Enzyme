#include "AliasAnalysis.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// The LLVM dialect is only used for the noalias attribute
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::dataflow;

static bool isPointerLike(Type type) {
  return isa<MemRefType, LLVM::LLVMPointerType>(type);
}

ChangeResult enzyme::AliasClassSet::join(const AliasClassSet &other) {
  if (unknown) {
    return ChangeResult::NoChange;
  }
  if (other.unknown) {
    unknown = true;
    return ChangeResult::Change;
  }

  return insert(other.aliasClasses);
}

ChangeResult
enzyme::AliasClassSet::insert(const DenseSet<DistinctAttr> &classes) {
  size_t oldSize = aliasClasses.size();
  aliasClasses.insert(classes.begin(), classes.end());
  return aliasClasses.size() == oldSize ? ChangeResult::NoChange
                                        : ChangeResult::Change;
}

ChangeResult enzyme::AliasClassSet::markFresh(Attribute debugLabel) {
  reset();

  auto freshClass = AliasClassLattice::getFresh(debugLabel);
  aliasClasses.insert(freshClass);
  return ChangeResult::Change;
}

ChangeResult enzyme::AliasClassSet::markUnknown() {
  if (unknown)
    return ChangeResult::NoChange;

  unknown = true;
  aliasClasses.clear();
  return ChangeResult::Change;
}

ChangeResult enzyme::AliasClassSet::reset() {
  if (aliasClasses.empty() && !unknown) {
    return ChangeResult::NoChange;
  }
  unknown = false;
  aliasClasses.clear();
  return ChangeResult::Change;
}

//===----------------------------------------------------------------------===//
// PointsToAnalysis
//===----------------------------------------------------------------------===//

template <typename T>
static ChangeResult mergeSets(DenseSet<T> &dest, const DenseSet<T> &src) {
  size_t oldSize = dest.size();
  dest.insert(src.begin(), src.end());
  return dest.size() == oldSize ? ChangeResult::NoChange : ChangeResult::Change;
}

void enzyme::PointsToSets::print(raw_ostream &os) const {
  for (const auto &[srcClass, destClasses] : pointsTo) {
    os << "  " << srcClass << " points to {";
    if (destClasses.isUnknown()) {
      os << "<unknown>";
    } else {
      llvm::interleaveComma(destClasses.getAliasClasses(), os);
    }
    os << "}\n";
  }
}

/// Union for every variable.
ChangeResult enzyme::PointsToSets::join(const AbstractDenseLattice &lattice) {
  const auto &rhs = static_cast<const PointsToSets &>(lattice);
  ChangeResult result = ChangeResult::NoChange;
  for (const auto &[alloc, allocSet] : rhs.pointsTo)
    result |= pointsTo[alloc].join(allocSet);
  return result;
}

/// Mark the pointer stored in `dest` as possibly pointing to any of `values`.
ChangeResult enzyme::PointsToSets::insert(DistinctAttr dest,
                                          const AliasClassSet &values) {
  return pointsTo[dest].join(values);
}

/// Mark the pointer stored in `dest` as possibly pointing to a fresh alias
/// class of values.
ChangeResult enzyme::PointsToSets::insertFresh(DistinctAttr dest,
                                               StringAttr debugLabel) {
  return pointsTo[dest].insert({AliasClassLattice::getFresh(debugLabel)});
}

ChangeResult enzyme::PointsToSets::markUnknown(DistinctAttr dest) {
  return pointsTo[dest].markUnknown();
}

ChangeResult enzyme::PointsToSets::markUnknown() {
  if (unknown)
    return ChangeResult::NoChange;

  unknown = true;
  pointsTo.clear();
  return ChangeResult::Change;
}

// TODO: Reduce code duplication with activity analysis
std::optional<Value> getStored(Operation *op);

void enzyme::PointsToPointerAnalysis::processCapturingStore(
    ProgramPoint dependent, PointsToSets *after, Value capturedValue,
    Value destinationAddress) {
  auto *srcClasses =
      getOrCreateFor<AliasClassLattice>(dependent, capturedValue);
  auto *destClasses =
      getOrCreateFor<AliasClassLattice>(dependent, destinationAddress);

  // If the destination class is unknown, i.e. all possible pointers, then we
  // have reached the pessimistic fixpoint and don't know anything. Bail.
  if (destClasses->isUnknown()) {
    propagateIfChanged(after, after->markUnknown());
    return;
  }

  for (DistinctAttr destClass : destClasses->getAliasClasses()) {
    // If the source class is unknown, note that any destination class may point
    // to any pointer.
    if (srcClasses->isUnknown())
      propagateIfChanged(after, after->markUnknown(destClass));
    else
      propagateIfChanged(
          after, after->insert(destClass, srcClasses->getAliasClassesObject()));
  }
}

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
        processCapturingStore(op, after, *stored, value);
      }
    }
  }
}

constexpr static llvm::StringLiteral kLLVMMemoryAttrName = "memory";

static std::pair<bool, bool>
isReadWriteOnly(FunctionOpInterface callee, unsigned argNo,
                std::optional<LLVM::ModRefInfo> argMemMRI) {
  unsigned numArguments = callee.getNumArguments();
  bool isReadOnly =
      (argMemMRI && *argMemMRI == LLVM::ModRefInfo::Ref) ||
      (argNo < numArguments &&
       !!callee.getArgAttr(argNo, LLVM::LLVMDialect::getReadonlyAttrName()));
  bool isWriteOnly =
      (argMemMRI && *argMemMRI == LLVM::ModRefInfo::Mod) ||
      (argNo < numArguments &&
       !!callee.getArgAttr(argNo, LLVM::LLVMDialect::getWriteOnlyAttrName()));
  assert(!(isReadOnly && isWriteOnly));
  return std::make_pair(isReadOnly, isWriteOnly);
}

static bool modRefMayMod(std::optional<LLVM::ModRefInfo> modRef) {
  return modRef ? (*modRef == LLVM::ModRefInfo::Mod ||
                   *modRef == LLVM::ModRefInfo::ModRef)
                : true;
}

static bool modRefMayRef(std::optional<LLVM::ModRefInfo> modRef) {
  return modRef ? (*modRef == LLVM::ModRefInfo::Ref ||
                   *modRef == LLVM::ModRefInfo::ModRef)
                : true;
}

void enzyme::PointsToPointerAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, CallControlFlowAction action,
    const PointsToSets &before, PointsToSets *after) {
  join(after, before);

  if (action == CallControlFlowAction::ExternalCallee) {
    // TODO(zinenko): this will fail an assertion for indirect calls. We should
    // be conservative here when we don't know the callee.
    auto symbol = cast<SymbolRefAttr>(call.getCallableForCallee());

    // Functions with known behavior.
    if (symbol.getLeafReference().getValue() == "posix_memalign") {
      // memalign deals with nested pointers and thus must be handled here
      // memalign points to a value
      OperandRange arguments = call.getArgOperands();
      auto *memPtr = getOrCreateFor<AliasClassLattice>(call, arguments[0]);
      for (DistinctAttr memPtrClass : memPtr->getAliasClasses()) {
        auto debugLabel = StringAttr::get(call.getContext(), "memalign");
        propagateIfChanged(after, after->insertFresh(memPtrClass, debugLabel));
      }
      return;
    }

    // Analyze the callee generically.
    if (auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
            call, symbol.getLeafReference())) {
      auto memoryAttr =
          callee->getAttrOfType<LLVM::MemoryEffectsAttr>(kLLVMMemoryAttrName);
      std::optional<LLVM::ModRefInfo> argModRef =
          memoryAttr ? std::make_optional(memoryAttr.getArgMem())
                     : std::nullopt;
      std::optional<LLVM::ModRefInfo> otherModRef =
          memoryAttr ? std::make_optional(memoryAttr.getOther()) : std::nullopt;

      // A function call may be capturing (storing) any pointers it takes into
      // any existing pointer, including those that are not directly passed as
      // arguments. The presence of specific attributes disables this behavior:
      //   - a pointer argument marked nocapture is not stored anywhere;
      //   - a pointer argument marked readonly is not stored into;
      //   - a function marked memory(arg: readonly) doesn't store anything
      //     into any argument pointer;
      //   - a function marked memory(other: readonly) doesn't store anything
      //     into pointers that are non-arguments.
      // TODO(zinenko): do we have a way to represent the conservative fixpoint,
      // e.g. a pointer can point to any other pointer? A _specific_ pointer
      // can point to any other pointer? How do we handle the case of "other
      // pointer" that has not been added to the set yet?

      SmallVector<int> pointerLikeOperands;
      for (auto &&[i, operand] : llvm::enumerate(call.getArgOperands())) {
        if (isPointerLike(operand.getType()))
          pointerLikeOperands.push_back(i);
      }

      // If the function may write to "other", that is any potential other
      // pointer, we can't reason anymore.
      // TODO(zinenko): is it possible to somehow say "other" may point to a
      // certain set of alias sets, or are we fully pessimistic here?
      bool funcMayWriteToOther = modRefMayMod(otherModRef);
      if (funcMayWriteToOther) {
        propagateIfChanged(after, after->markUnknown());
        return;
      }

      // If the function may read from "other", it may be storing any pointer
      // into the arguments.
      bool funcMayReadOther = modRefMayRef(otherModRef);
      if (funcMayReadOther) {
        for (int pointerAsAddress : pointerLikeOperands) {
          auto *destClasses = getOrCreateFor<AliasClassLattice>(
              call, call.getArgOperands()[pointerAsAddress]);
          // TODO: fold this into lattice method and have only one change
          // result.
          for (DistinctAttr destClass : destClasses->getAliasClasses())
            propagateIfChanged(after, after->markUnknown(destClass));
        }
        return;
      }

      unsigned numArguments = callee.getNumArguments();
      bool funcMayWriteToArgs = modRefMayMod(argModRef);
      for (int pointerAsData : pointerLikeOperands) {
        // If not captured, it cannot be stored in anything.
        bool isCaptured =
            (pointerAsData < numArguments &&
             !callee.getArgAttr(pointerAsData,
                                LLVM::LLVMDialect::getNoCaptureAttrName()));
        if (!isCaptured)
          continue;

        for (int pointerAsAddress : pointerLikeOperands) {
          // If cannot store into, nothing to do here.
          auto [isReadOnly, isWriteOnly] =
              isReadWriteOnly(callee, pointerAsAddress, argModRef);
          bool isStoredInto =
              (!isReadOnly || isWriteOnly) && funcMayWriteToArgs;
          if (!isStoredInto)
            continue;

          processCapturingStore(call, after,
                                call.getArgOperands()[pointerAsData],
                                call.getArgOperands()[pointerAsAddress]);
        }
      }

      // TODO(zinenko): handle noalias and other things for results

      return;
    }
    // TODO(zinenko): should we be more conservative here when we couldn't
    // process the function? Looks related to "unimplemented unknown" and the
    // absence of fixpoint status?
  }
}

// The default initialization (empty map of empty sets) is correct.
void enzyme::PointsToPointerAnalysis::setToEntryState(PointsToSets *lattice) {}

//===----------------------------------------------------------------------===//
// AliasClassLattice
//===----------------------------------------------------------------------===//

void enzyme::AliasClassLattice::print(raw_ostream &os) const {
  if (aliasClasses.isUnknown()) {
    os << "Unknown AC";
  } else {
    os << "size: " << aliasClasses.getAliasClasses().size() << ":\n";
    for (auto aliasClass : aliasClasses.getAliasClasses()) {
      os << "  " << aliasClass << "\n";
    }
  }
}

AliasResult
enzyme::AliasClassLattice::alias(const AbstractSparseLattice &other) const {
  const auto *rhs = reinterpret_cast<const AliasClassLattice *>(&other);
  if (getPoint() == rhs->getPoint())
    return AliasResult::MustAlias;

  if (aliasClasses.isUnknown() || rhs->aliasClasses.isUnknown())
    return AliasResult::MayAlias;

  size_t overlap = llvm::count_if(
      aliasClasses.getAliasClasses(), [rhs](DistinctAttr aliasClass) {
        return rhs->aliasClasses.getAliasClasses().contains(aliasClass);
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
  return aliasClasses.join(otherAliasClass->aliasClasses);
}

ChangeResult enzyme::AliasClassLattice::markFresh(Attribute debugLabel) {
  reset();

  Value value = getPoint();
  if (!debugLabel)
    debugLabel = UnitAttr::get(value.getContext());
  return aliasClasses.markFresh(debugLabel);
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
        if (isPointerLike(arg.getType()))
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
      for (auto srcClass : getLatticeElement(value)->getAliasClasses()) {
        const auto &srcPointsTo = pointsToSets->pointsTo.lookup(srcClass);
        for (AliasClassLattice *result : results) {
          propagateIfChanged(result,
                             result->insert(srcPointsTo.getAliasClasses()));
        }
      }
    }
  }

  // TODO(zinenko): this is sketchy. We could have an operation that has side
  // effects and loads a pointer from another pointer, but also has another
  // result that aliases the operand. So returning here is premature.
  if (!effects.empty())
    return;

  // For operations that don't touch memory, conservatively assume all results
  // alias all operands.
  for (auto *resultLattice : results) {
    for (const auto *operandLattice : operands) {
      join(resultLattice, *operandLattice);
    }
  }
}

void populateConservativeCallEffects(
    CallOpInterface call,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (Value argument : call.getArgOperands()) {
    if (!isPointerLike(argument.getType()))
      continue;

    effects.emplace_back(MemoryEffects::Read::get(), argument);
    effects.emplace_back(MemoryEffects::Write::get(), argument);
    // TODO: consider having a may-free effect.
  }
}

// TODO: Move this somewhere shared
void getEffectsForExternalCall(
    CallOpInterface call,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  if (!symbol)
    return populateConservativeCallEffects(call, effects);

  // Functions with known specific behavior.
  StringRef callableName = symbol.getLeafReference().getValue();
  if (callableName == "malloc" || callableName == "_Znwm") {
    assert(call->getNumResults() == 1);
    effects.push_back(MemoryEffects::EffectInstance(
        MemoryEffects::Allocate::get(), call->getResult(0)));
    return;
  }

  // Generic reasoning based on attributes.
  auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      call, symbol.getLeafReference());
  if (!callee)
    return populateConservativeCallEffects(call, effects);

  // A function by default has all possible effects on all pointer-like
  // arguments. Presence of specific attributes removes those effects.
  auto memoryAttr =
      callee->getAttrOfType<LLVM::MemoryEffectsAttr>(kLLVMMemoryAttrName);
  std::optional<LLVM::ModRefInfo> argMemMRI =
      memoryAttr ? std::make_optional(memoryAttr.getArgMem()) : std::nullopt;

  unsigned numArguments = callee.getNumArguments();
  for (auto &&[i, argument] : llvm::enumerate(call.getArgOperands())) {
    if (!isPointerLike(argument.getType()))
      continue;

    bool isReadNone =
        (argMemMRI && *argMemMRI == LLVM::ModRefInfo::NoModRef) ||
        (i < numArguments &&
         !!callee.getArgAttr(i, LLVM::LLVMDialect::getReadnoneAttrName()));
    auto [isReadOnly, isWriteOnly] = isReadWriteOnly(callee, i, argMemMRI);

    if (!isReadNone && !isWriteOnly)
      effects.emplace_back(MemoryEffects::Read::get(), argument);

    if (!isReadOnly)
      effects.emplace_back(MemoryEffects::Write::get(), argument);

    // TODO: consider having a may-free effect.
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

void enzyme::AliasAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const AliasClassLattice *> operands,
    ArrayRef<AliasClassLattice *> results) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  getEffectsForExternalCall(call, effects);
  transfer(call, effects, operands, results);
}
