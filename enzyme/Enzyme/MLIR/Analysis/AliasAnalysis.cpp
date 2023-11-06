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

// The LLVM dialect is only used for attribute names.
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
  if (unknown)
    return ChangeResult::NoChange;

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
  if (unknown) {
    os << "<fully-unknown>\n";
    return;
  }

  if (pointsTo.empty()) {
    os << "<empty>\n";
    return;
  }

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
  if (unknown)
    return ChangeResult::NoChange;

  if (rhs.unknown) {
    unknown = true;
    return ChangeResult::Change;
  }

  ChangeResult result = ChangeResult::NoChange;
  for (const auto &[alloc, allocSet] : rhs.pointsTo)
    result |= pointsTo[alloc].join(allocSet);
  return result;
}

/// Mark the pointer stored in `dest` as possibly pointing to any of `values`.
ChangeResult enzyme::PointsToSets::insert(const AliasClassSet &destClasses,
                                          const AliasClassSet &values) {
  if (unknown)
    return ChangeResult::NoChange;

  ChangeResult result = ChangeResult::NoChange;
  for (auto destClass : destClasses.getAliasClasses())
    result |= pointsTo[destClass].insert(values.getAliasClasses());
  return result;
}

/// Mark the pointer stored in `dest` as possibly pointing to a fresh alias
/// class of values.
ChangeResult enzyme::PointsToSets::insertFresh(DistinctAttr dest,
                                               StringAttr debugLabel) {
  // TODO(zinenko): do we need some sort of "exact/inexact" in addition to this?
  // i.e. can we go back from full-unknown state to knowing that some pointers
  // point to specific aliases classes, and other pointers (potentially not
  // present in the list) may be pointing to anything?
  if (unknown)
    return ChangeResult::NoChange;

  return pointsTo[dest].insert({AliasClassLattice::getFresh(debugLabel)});
}

ChangeResult
enzyme::PointsToSets::addSetsFrom(const AliasClassSet &destClasses,
                                  const AliasClassSet &srcClasses) {
  if (unknown)
    return ChangeResult::NoChange;

  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr destClass : destClasses.getAliasClasses()) {
    for (DistinctAttr srcClass : srcClasses.getAliasClasses()) {
      result |= pointsTo[destClass].join(pointsTo[srcClass]);
    }
  }
  return result;
}

ChangeResult
enzyme::PointsToSets::setToFresh(const AliasClassSet &destClasses) {
  // TODO(zinenko): we may want to override this, for a specific pointer we know
  // it points to fresh and all the other points are "unknown". Currently there
  // is no way of expressing this.
  if (unknown)
    return ChangeResult::NoChange;

  DistinctAttr fresh = AliasClassLattice::getFresh(StringAttr::get(
      getPoint().getLoc()->getContext(), "function-result-noalias"));
  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr destClass : destClasses.getAliasClasses()) {
    result |= pointsTo[destClass].reset();
    result |= pointsTo[destClass].insert({fresh});
  }
  return result;
}

ChangeResult
enzyme::PointsToSets::markUnknown(const AliasClassSet &destClasses) {
  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr destClass : destClasses.getAliasClasses())
    result |= pointsTo[destClass].markUnknown();
  return result;
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

  // If the source class is unknown, record that any destination class may
  // point to any pointer.
  if (srcClasses->isUnknown()) {
    propagateIfChanged(
        after, after->markUnknown(destClasses->getAliasClassesObject()));
  } else {
    propagateIfChanged(after,
                       after->insert(destClasses->getAliasClassesObject(),
                                     srcClasses->getAliasClassesObject()));
  }
}

void enzyme::PointsToPointerAnalysis::visitOperation(Operation *op,
                                                     const PointsToSets &before,
                                                     PointsToSets *after) {
  using llvm::errs;
  join(after, before);

  // If we know nothing about memory effects, record reaching the pessimistic
  // fixpoint and bail.
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    after->markUnknown();
    return;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;

    SmallVector<Value> targetValues;
    Value value = effect.getValue();
    auto pointerLikeOperands =
        llvm::make_filter_range(op->getOperands(), [](Value operand) {
          return isPointerLike(operand.getType());
        });

    // If we don't know the value on which the effect happens, it can happen on
    // any value.
    if (value) {
      targetValues.push_back(value);
    } else {
      llvm::append_range(targetValues, pointerLikeOperands);
    }

    // If we don't know which value is stored, it can be any value. For the
    // purpose of this analysis, we only care about pointer-like values being
    // stored.
    SmallVector<Value> storedValues;
    if (std::optional<Value> stored = getStored(op)) {
      storedValues.push_back(*stored);
    } else {
      llvm::append_range(storedValues, pointerLikeOperands);
    }

    for (Value address : targetValues) {
      for (Value stored : storedValues) {
        processCapturingStore(op, after, stored, address);
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
    // When we don't know the callee, be conservative.
    // TODO: eventually consider an additional "points-to" analysis for indirect
    // calls.
    auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
    if (!symbol)
      return propagateIfChanged(after, after->markUnknown());

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
    // A function call may be capturing (storing) any pointers it takes into
    // any existing pointer, including those that are not directly passed as
    // arguments. The presence of specific attributes restricts this behavior:
    //   - a pointer argument marked nocapture is not stored anywhere;
    //   - a pointer argument marked readonly is not stored into;
    //   - a function marked memory(arg: readonly) doesn't store anything
    //     into any argument pointer;
    //   - a function marked memory(other: readonly) doesn't store anything
    //     into pointers that are non-arguments.
    if (auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
            call, symbol.getLeafReference())) {
      auto memoryAttr =
          callee->getAttrOfType<LLVM::MemoryEffectsAttr>(kLLVMMemoryAttrName);
      std::optional<LLVM::ModRefInfo> argModRef =
          memoryAttr ? std::make_optional(memoryAttr.getArgMem())
                     : std::nullopt;
      std::optional<LLVM::ModRefInfo> otherModRef =
          memoryAttr ? std::make_optional(memoryAttr.getOther()) : std::nullopt;
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
      // into the arguments. At this point, we know it shouldn't be also writing
      // to "other".
      bool funcMayReadOther = modRefMayRef(otherModRef);
      bool funcMayWriteToArgs = modRefMayMod(argModRef);
      auto mayArgBeStoredInto = [&](int arg) {
        auto [isReadOnly, isWriteOnly] =
            isReadWriteOnly(callee, arg, argModRef);
        return (!isReadOnly || isWriteOnly) && funcMayWriteToArgs;
      };
      unsigned numArguments = callee.getNumArguments();
      if (funcMayReadOther) {
        for (int pointerAsAddress : pointerLikeOperands) {
          if (!mayArgBeStoredInto(pointerAsAddress))
            continue;

          auto *destClasses = getOrCreateFor<AliasClassLattice>(
              call, call.getArgOperands()[pointerAsAddress]);
          propagateIfChanged(
              after, after->markUnknown(destClasses->getAliasClassesObject()));
        }
      } else {
        for (int pointerAsData : pointerLikeOperands) {
          // If not captured, it cannot be stored in anything. However, another
          // pointer that may be stored in this pointer can be stored in another
          // writable pointer.
          bool isNoCapture =
              (pointerAsData < numArguments &&
               !!callee.getArgAttr(pointerAsData,
                                   LLVM::LLVMDialect::getNoCaptureAttrName()));

          for (int pointerAsAddress : pointerLikeOperands) {
            if (!mayArgBeStoredInto(pointerAsAddress))
              continue;

            // In all cases, we want to record that "destination" may be
            // pointing to same classes as "source".
            const auto *srcClasses = getOrCreateFor<AliasClassLattice>(
                call, call.getArgOperands()[pointerAsData]);
            const auto *destClasses = getOrCreateFor<AliasClassLattice>(
                call, call.getArgOperands()[pointerAsAddress]);
            propagateIfChanged(
                after, after->addSetsFrom(destClasses->getAliasClassesObject(),
                                          srcClasses->getAliasClassesObject()));

            // If "source" is also captured, then "destination" may additionally
            // be pointing to the classes of "source" itself.
            if (isNoCapture)
              continue;
            processCapturingStore(call, after,
                                  call.getArgOperands()[pointerAsData],
                                  call.getArgOperands()[pointerAsAddress]);
          }
        }
      }

      // Pointer-typed results may be pointing to any other pointer. The
      // presence of attributes restricts this behavior:
      //   - If the function is marked memory(...) so that it doesn't read from
      //     "other" memory, the return values may be pointing only to
      //     same alias classes as arguments + arguments themselves + a new
      //     alias class for a potential allocation.
      //   - Additionally, if any of the arguments are annotated as writeonly,
      //     the results should not point to alias classes those arguments are
      //     pointing to.
      //   - Additionally, if any of the arguments are annotated as nocapture,
      //     the results should not point to those arguments themselves.
      //   - If the function is marked as not reading from arguments, the
      //     results should not point to any alias classes pointed to by the
      //     arguments.
      bool funcMayReadArgs = modRefMayRef(argModRef);
      for (OpResult result : call->getResults()) {
        if (!isPointerLike(result.getType()))
          continue;

        const auto *destClasses =
            getOrCreateFor<AliasClassLattice>(call, result);

        // If reading from other memory, the results may point to anything.
        if (funcMayReadOther) {
          propagateIfChanged(
              after, after->markUnknown(destClasses->getAliasClassesObject()));
          continue;
        }

        ChangeResult changed = ChangeResult::NoChange;
        AliasClassSet commonReturnScope;
        (void)commonReturnScope.markFresh(
            StringAttr::get(call->getContext(), "function-return-common"));
        for (int operandNo : pointerLikeOperands) {
          const auto *srcClasses = getOrCreateFor<AliasClassLattice>(
              call, call.getArgOperands()[operandNo]);
          bool maybeRead =
              funcMayReadArgs &&
              (operandNo >= numArguments ||
               !callee.getArgAttr(operandNo,
                                  LLVM::LLVMDialect::getWriteOnlyAttrName()));
          if (maybeRead) {
            changed |= after->addSetsFrom(destClasses->getAliasClassesObject(),
                                          srcClasses->getAliasClassesObject());
          }

          bool isNoCapture =
              (operandNo < numArguments &&
               !!callee.getArgAttr(operandNo,
                                   LLVM::LLVMDialect::getNoCaptureAttrName()));
          if (isNoCapture) {
            changed |= after->insert(destClasses->getAliasClassesObject(),
                                     srcClasses->getAliasClassesObject());
          }
          after->insert(destClasses->getAliasClassesObject(),
                        commonReturnScope);
        }
        propagateIfChanged(after, changed);
      }

      return;
    }

    // Don't know how to handle, record pessimistic fixpoint.
    return propagateIfChanged(after, after->markUnknown());
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

  // TODO: handle this case properly.
  //   - If the results are marked "noalias", they may _only_ point to a
  //     fresh alias scope as per the LLVM specification of "noalias" on
  //     results.
  // Noalias result is pointing to a fresh aliasing class.
  // if (callee.getResultAttr(result.getResultNumber(),
  //                          LLVM::LLVMDialect::getNoAliasAttrName())) {
  //   propagateIfChanged(
  //       after, after->setToFresh(destClasses->getAliasClassesObject()));
  //   continue;
  // }
}
