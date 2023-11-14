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

// TODO: remove this once aliasing interface is factored out.
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::dataflow;

static bool isPointerLike(Type type) {
  return isa<MemRefType, LLVM::LLVMPointerType>(type);
}

const enzyme::AliasClassSet enzyme::AliasClassSet::emptySet = AliasClassSet();
const enzyme::AliasClassSet enzyme::AliasClassSet::unknownSet =
    AliasClassSet(true);

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

bool enzyme::AliasClassSet::isCanonical() const {
  return !unknown || aliasClasses.empty();
}

bool enzyme::AliasClassSet::operator==(
    const enzyme::AliasClassSet &other) const {
  assert(isCanonical() && other.isCanonical());
  return unknown == other.unknown &&
         llvm::equal(aliasClasses, other.aliasClasses);
}

ChangeResult enzyme::AliasClassSet::foreachClass(
    function_ref<ChangeResult(DistinctAttr)> callback) const {
  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr attr : aliasClasses)
    result |= callback(attr);
  return result;
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
  os << "other points to unknown: " << otherPointToUnknown << "\n";
}

/// Union for every variable.
ChangeResult enzyme::PointsToSets::join(const AbstractDenseLattice &lattice) {
  const auto &rhs = static_cast<const PointsToSets &>(lattice);

  // Both are exact, just join and carry over pointer classes from RHS.
  if (!otherPointToUnknown && !rhs.otherPointToUnknown) {
    ChangeResult result = ChangeResult::NoChange;
    for (const auto &[otherPointer, otherPointee] : rhs.pointsTo) {
      result |= pointsTo[otherPointer].join(otherPointee);
    }
    return result;
  }

  // If this has other pointers pointing to unknown, only join in the RHS
  // pointers that are known on the LHS. If some LHS pointers are not present in
  // RHS, keep them as is because RHS is "exact".
  if (otherPointToUnknown && !rhs.otherPointToUnknown) {
    ChangeResult result = ChangeResult::NoChange;
    for (DistinctAttr pointer : llvm::make_first_range(pointsTo)) {
      auto it = rhs.pointsTo.find(pointer);
      if (it != rhs.pointsTo.end())
        result |= pointsTo[pointer].join(it->getSecond());
    }
    return result;
  }

  // If both have other pointers pointing to unknown, only join the pointers
  // that are present simultaneously in LHS and RHS. Drop LHS pointers that
  // are not present in RHS from the list (they would explicitly point to
  // unknown on individual join, but this is implied by the otherPointsToUnknown
  // flag). Create a temporary vector for iteration as we will be erasing from
  // the map in the loop.
  if (otherPointToUnknown && rhs.otherPointToUnknown) {
    ChangeResult result = ChangeResult::NoChange;
    for (DistinctAttr pointer :
         llvm::to_vector(llvm::make_first_range(pointsTo))) {
      auto it = rhs.pointsTo.find(pointer);
      if (it != rhs.pointsTo.end()) {
        result |= pointsTo[pointer].join(it->getSecond());
      } else {
        pointsTo.erase(pointer);
        result = ChangeResult::Change;
      }
    }
    return result;
  }

  // If RHS has other pointers pointing to unknown, only join the pointers that
  // are present in both simultaneously. Drop LHS pointers that are not present
  // in RHS (they would explicitly point to unknown on individual join but this
  // is implied by the otherPointsToUnknown flag). Set RHS to also indicate
  // other pointers pointing to unknown.
  assert(!otherPointToUnknown && rhs.otherPointToUnknown);
  otherPointToUnknown = true;
  for (DistinctAttr pointer :
       llvm::to_vector(llvm::make_first_range(pointsTo))) {
    auto it = rhs.pointsTo.find(pointer);
    if (it != rhs.pointsTo.end())
      pointsTo[pointer].join(rhs.getPointsTo(pointer));
    else
      pointsTo.erase(pointer);
  }
  return ChangeResult::Change;
}

ChangeResult enzyme::PointsToSets::update(const AliasClassSet &keysToUpdate,
                                          const AliasClassSet &values,
                                          bool replace) {
  // If updating the unknown alias class to point to something, we have reached
  // the pessimistic fixpoint.
  if (keysToUpdate.isUnknown())
    return markAllPointToUnknown();

  // If updating to point to unknown, and we already know others are pointing to
  // unknown, just erase the known information.
  if (values.isUnknown() && otherPointToUnknown) {
    return keysToUpdate.foreachClass([&](DistinctAttr dest) {
      return pointsTo.erase(dest) ? ChangeResult::Change
                                  : ChangeResult::NoChange;
    });
  }

  // Otherwise just set the result.
  if (replace) {
    return keysToUpdate.foreachClass([&](DistinctAttr dest) {
      auto it = pointsTo.find(dest);
      if (it != pointsTo.end() && it->getSecond() == values)
        return ChangeResult::NoChange;
      if (it == pointsTo.end())
        pointsTo.try_emplace(dest, values);
      else
        it->second = values;
      return ChangeResult::Change;
    });
  }

  return keysToUpdate.foreachClass([&](DistinctAttr dest) {
    // If pointers stored in "other" are pointing to unknown alias class, don't
    // override that.
    if (otherPointToUnknown && !pointsTo.count(dest))
      return ChangeResult::NoChange;

    if (values.isUnknown())
      return pointsTo[dest].markUnknown();
    return pointsTo[dest].insert(values.getAliasClasses());
  });
}

ChangeResult
enzyme::PointsToSets::setPointingToFresh(const AliasClassSet &destClasses,
                                         StringAttr debugLabel) {
  return update(destClasses, AliasClassLattice::getFresh(debugLabel),
                /*replace=*/true);
}

ChangeResult
enzyme::PointsToSets::addSetsFrom(const AliasClassSet &destClasses,
                                  const AliasClassSet &srcClasses) {
  if (destClasses.isUnknown())
    return markAllPointToUnknown();

  return destClasses.foreachClass([&](DistinctAttr dest) {
    return srcClasses.foreachClass(
        [&](DistinctAttr src) { return pointsTo[dest].join(pointsTo[src]); });
  });
}

ChangeResult
enzyme::PointsToSets::markPointToUnknown(const AliasClassSet &destClasses) {
  if (destClasses.isUnknown())
    return markAllPointToUnknown();

  return destClasses.foreachClass(
      [&](DistinctAttr dest) { return pointsTo[dest].markUnknown(); });
}

ChangeResult enzyme::PointsToSets::markAllPointToUnknown() {
  if (otherPointToUnknown && pointsTo.empty())
    return ChangeResult::NoChange;

  otherPointToUnknown = true;
  pointsTo.clear();
  return ChangeResult::Change;
}

ChangeResult enzyme::PointsToSets::markAllExceptPointToUnknown(
    const AliasClassSet &destClasses) {
  bool wasOtherPointingToUnknown = otherPointToUnknown;
  otherPointToUnknown = true;

  llvm::SmallDenseSet<DistinctAttr, 8> keysToDelete;
  for (DistinctAttr key : llvm::make_first_range(pointsTo)) {
    if (!destClasses.getAliasClasses().contains(key))
      keysToDelete.insert(key);
  }
  for (DistinctAttr key : keysToDelete)
    pointsTo.erase(key);
  return (wasOtherPointingToUnknown && keysToDelete.empty())
             ? ChangeResult::NoChange
             : ChangeResult::Change;
}

// TODO: Reduce code duplication with activity analysis
std::optional<Value> getStored(Operation *op);

void enzyme::PointsToPointerAnalysis::processCapturingStore(
    ProgramPoint dependent, PointsToSets *after, Value capturedValue,
    Value destinationAddress, bool isMustStore) {
  auto *srcClasses =
      getOrCreateFor<AliasClassLattice>(dependent, capturedValue);
  auto *destClasses =
      getOrCreateFor<AliasClassLattice>(dependent, destinationAddress);

  // If the destination class is unknown, i.e. all possible pointers, then we
  // have reached the pessimistic fixpoint and don't know anything. Bail.
  if (destClasses->isUnknown()) {
    propagateIfChanged(after, after->markAllPointToUnknown());
    return;
  }

  // If the source class is unknown, record that any destination class may
  // point to any pointer.
  if (srcClasses->isUnknown()) {
    propagateIfChanged(
        after, after->markPointToUnknown(destClasses->getAliasClassesObject()));
  } else {
    // Treat all stores as may-store because we don't know better.
    if (isMustStore) {
      propagateIfChanged(after, after->setPointingToClasses(
                                    destClasses->getAliasClassesObject(),
                                    srcClasses->getAliasClassesObject()));
    } else {
      propagateIfChanged(after,
                         after->insert(destClasses->getAliasClassesObject(),
                                       srcClasses->getAliasClassesObject()));
    }
  }
}

// TODO: this should become an interface or be integrated into side effects so
// it doesn't depend on the dialect.
// TODO: currently, we should treat all stores is "may store" because of how the
// analysis is used: the points-to of the function exit point is considered as
// the overall state of the function, which may be incorrect for any operation
// post-dominated by a must-store.
static bool isMustStore(Operation *op, Value pointer) {
  return false; // isa<LLVM::StoreOp>(op);
}

void enzyme::PointsToPointerAnalysis::visitOperation(Operation *op,
                                                     const PointsToSets &before,
                                                     PointsToSets *after) {
  join(after, before);

  // If we know nothing about memory effects, record reaching the pessimistic
  // fixpoint and bail.
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    propagateIfChanged(after, after->markAllPointToUnknown());
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
        processCapturingStore(op, after, stored, address,
                              isMustStore(op, address));
      }
    }
  }
}

constexpr static llvm::StringLiteral kLLVMMemoryAttrName = "memory";

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

static bool mayReadArg(FunctionOpInterface callee, unsigned argNo,
                       std::optional<LLVM::ModRefInfo> argMemMRI) {
  // Function-wide annotation.
  bool funcMayRead = modRefMayRef(argMemMRI);

  // Vararg behavior can only be specified by the function.
  unsigned numArguments = callee.getNumArguments();
  if (argNo >= numArguments)
    return funcMayRead;

  bool hasWriteOnlyAttr =
      !!callee.getArgAttr(argNo, LLVM::LLVMDialect::getWriteOnlyAttrName());
  bool hasReadNoneAttr =
      !!callee.getArgAttr(argNo, LLVM::LLVMDialect::getReadnoneAttrName());
  return !hasWriteOnlyAttr && !hasReadNoneAttr && funcMayRead;
}

static bool mayWriteArg(FunctionOpInterface callee, unsigned argNo,
                        std::optional<LLVM::ModRefInfo> argMemMRI) {
  // Function-wide annotation.
  bool funcMayWrite = modRefMayMod(argMemMRI);

  // Vararg behavior can only be specified by the function.
  unsigned numArguments = callee.getNumArguments();
  if (argNo >= numArguments)
    return funcMayWrite;

  // Individual attributes can further restrict argument writability. Note that
  // `readnone` means "no read or write" for LLVM.
  bool hasReadOnlyAttr =
      !!callee.getArgAttr(argNo, LLVM::LLVMDialect::getReadonlyAttrName());
  bool hasReadNoneAttr =
      !!callee.getArgAttr(argNo, LLVM::LLVMDialect::getReadnoneAttrName());
  return !hasReadOnlyAttr && !hasReadNoneAttr && funcMayWrite;
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
      return propagateIfChanged(after, after->markAllPointToUnknown());

    // Functions with known behavior.
    if (symbol.getLeafReference().getValue() == "posix_memalign") {
      // memalign deals with nested pointers and thus must be handled here
      // memalign points to a value
      OperandRange arguments = call.getArgOperands();
      auto *memPtr = getOrCreateFor<AliasClassLattice>(call, arguments[0]);
      for (DistinctAttr memPtrClass : memPtr->getAliasClasses()) {
        // Note that this is a "must write" kind of situation, so we can
        // directly set the classes pointed to, rather than inserting them.
        auto debugLabel = StringAttr::get(call.getContext(), "memalign");
        propagateIfChanged(after,
                           after->setPointingToFresh(memPtrClass, debugLabel));
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

      // Precompute the set of alias classes the function may capture.
      // TODO: consider a more advanced lattice that can encode "may point to
      // any class _except_ the given classes"; this is mathematically possible
      // but needs careful programmatic encoding.
      AliasClassSet functionMayCapture;
      bool funcMayReadOther = modRefMayRef(otherModRef);
      unsigned numArguments = callee.getNumArguments();
      if (funcMayReadOther) {
        // If a function may read from other, it may be storing pointers from
        // unknown alias sets into any writable pointer.
        functionMayCapture.markUnknown();
      } else {
        for (int pointerAsData : pointerLikeOperands) {
          // If not captured, it cannot be stored in anything.
          if ((pointerAsData < numArguments &&
               !!callee.getArgAttr(pointerAsData,
                                   LLVM::LLVMDialect::getNoCaptureAttrName())))
            continue;

          const auto *srcClasses = getOrCreateFor<AliasClassLattice>(
              call, call.getArgOperands()[pointerAsData]);
          functionMayCapture.join(srcClasses->getAliasClassesObject());
        }
      }

      AliasClassSet pointerOperandClasses;
      ChangeResult changed = ChangeResult::NoChange;
      for (int pointerOperand : pointerLikeOperands) {
        auto *destClasses = getOrCreateFor<AliasClassLattice>(
            call, call.getArgOperands()[pointerOperand]);
        pointerOperandClasses.join(destClasses->getAliasClassesObject());

        // If the argument cannot be stored into, just preserve it as is.
        if (!mayWriteArg(callee, pointerOperand, argModRef))
          continue;

        // If the destination class is unknown, we reached the pessimistic
        // fixpoint.
        if (destClasses->isUnknown()) {
          pointerOperandClasses.reset();
          changed |= after->markAllPointToUnknown();
          break;
        }

        // Otherwise, indicate that a pointer that belongs to any of the
        // classes captured by this function may be stored into the
        // destination class.
        changed |= destClasses->getAliasClassesObject().foreachClass(
            [&](DistinctAttr dest) {
              return after->insert(dest, functionMayCapture);
            });
      }

      // If the function may write to "other", that is any potential other
      // pointer, record that.
      if (modRefMayMod(otherModRef)) {
        // All other alias classes that are not present as arguments should
        // point to unknown.
        // Since:
        //  - `after` was joined with `before` at the beginning; and
        //  - pre-existing keys in `after` (and in `before` since no new keys
        //    were added) have their values: preserved, joined with another
        //    alias set (->insert is a join), or removed here with default value
        //    being set to "any" (lattice top);
        // this transfer function is monotonic with respect to its input, i.e,
        // the `before` lattice.
        // TODO(zinenko): consider monotonicity more carefully wrt to
        // `destClasses` change.
        changed |= after->markAllExceptPointToUnknown(pointerOperandClasses);
      }

      // Pointer-typed results may be pointing to any other pointer. The
      // presence of attributes restricts this behavior:
      //   - If the function is marked memory(...) so that it doesn't read from
      //     "other" memory, the return values may be pointing only to
      //     same alias classes as arguments + arguments themselves + a new
      //     alias class for a potential allocation.
      //   - Additionally, if any of the arguments are annotated as writeonly or
      //     readnone, the results should not point to alias classes those
      //     arguments are pointing to.
      //   - Additionally, if any of the arguments are annotated as nocapture,
      //     the results should not point to those arguments themselves.
      //   - If the function is marked as not reading from arguments, the
      //     results should not point to any alias classes pointed to by the
      //     arguments.
      for (OpResult result : call->getResults()) {
        if (!isPointerLike(result.getType()))
          continue;

        const auto *destClasses =
            getOrCreateFor<AliasClassLattice>(call, result);

        // If reading from other memory, the results may point to anything.
        if (funcMayReadOther) {
          propagateIfChanged(after, after->markPointToUnknown(
                                        destClasses->getAliasClassesObject()));
          continue;
        }

        AliasClassSet commonReturnScope;
        (void)commonReturnScope.markFresh(
            StringAttr::get(call->getContext(), "function-return-common"));
        for (int operandNo : pointerLikeOperands) {
          const auto *srcClasses = getOrCreateFor<AliasClassLattice>(
              call, call.getArgOperands()[operandNo]);
          if (mayReadArg(callee, operandNo, argModRef)) {
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
      }
      return propagateIfChanged(after, changed);
    }

    // Don't know how to handle, record pessimistic fixpoint.
    return propagateIfChanged(after, after->markAllPointToUnknown());
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

/// Returns `true` if the alias transfer function of the operation is fully
/// described by its memory effects.
//
// We could have an operation that has side effects and loads a pointer from
// another pointer, but also has another result that aliases the operand, which
// would need additional processing.
//
// TODO: turn this into an interface.
static bool isAliasTransferFullyDescribedByMemoryEffects(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, LLVM::LoadOp, LLVM::StoreOp>(op);
}

void enzyme::AliasAnalysis::transfer(
    Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects,
    ArrayRef<const AliasClassLattice *> operands,
    ArrayRef<AliasClassLattice *> results) {
  bool globalRead = false;
  for (const auto &effect : effects) {
    // If the effect is global read, record that.
    Value value = effect.getValue();
    if (!value) {
      globalRead |= isa<MemoryEffects::Read>(effect.getEffect());
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
      AliasClassLattice *latticeElement = getLatticeElement(value);
      if (latticeElement->isUnknown()) {
        for (AliasClassLattice *result : results) {
          propagateIfChanged(result, result->markUnknown());
        }
      } else {
        for (auto srcClass : latticeElement->getAliasClasses()) {
          const auto &srcPointsTo = pointsToSets->getPointsTo(srcClass);
          for (AliasClassLattice *result : results) {
            // TODO: consider some sort of "point join" or better insert that
            // doesn't require a conditional here.
            if (srcPointsTo.isUnknown()) {
              propagateIfChanged(result, result->markUnknown());
            } else {
              // TODO: this looks potentially non-monotonous.
              ChangeResult r = result->reset() |
                               result->insert(srcPointsTo.getAliasClasses());
              propagateIfChanged(result, r);
            }
          }
        }
      }
    }
  }

  // If there was a global read effect, the operation may be reading from any
  // pointer so we cannot say what the results are pointing to. Can safely exit
  // here because all results are now in the fixpoint state.
  if (globalRead) {
    for (auto *resultLattice : results)
      propagateIfChanged(resultLattice, resultLattice->markUnknown());
    return;
  }

  // If it was enough to reason about effects, exit here.
  if (!effects.empty() && isAliasTransferFullyDescribedByMemoryEffects(op))
    return;

  // Conservatively assume all results alias all operands.
  for (AliasClassLattice *resultLattice : results) {
    ChangeResult r = ChangeResult::NoChange;
    for (const AliasClassLattice *operandLattice : operands)
      r |= resultLattice->join(*operandLattice);
    propagateIfChanged(resultLattice, r);
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
LogicalResult getEffectsForExternalCall(
    CallOpInterface call,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  if (!symbol)
    return failure();

  // Functions with known specific behavior.
  StringRef callableName = symbol.getLeafReference().getValue();
  if (callableName == "malloc" || callableName == "_Znwm") {
    assert(call->getNumResults() == 1);
    effects.push_back(MemoryEffects::EffectInstance(
        MemoryEffects::Allocate::get(), call->getResult(0)));
    return success();
  }

  return failure();
}

void enzyme::AliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const AliasClassLattice *> operands,
    ArrayRef<AliasClassLattice *> results) {

  // If we don't have memory effect information, don't assume anything about
  // values.
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    for (OpResult result : op->getResults()) {
      if (!isPointerLike(result.getType()))
        continue;

      results[result.getResultNumber()]->markUnknown();
    }
    return;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
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
  // First, try effect-based reasoning for known functions.
  SmallVector<MemoryEffects::EffectInstance> effects;
  if (succeeded(getEffectsForExternalCall(call, effects)))
    return transfer(call, effects, operands, results);

  auto markResultsUnknown = [&] {
    for (AliasClassLattice *resultLattice : results)
      propagateIfChanged(resultLattice, resultLattice->markUnknown());
  };

  // If failed, try using function attributes. If results are marked noalias,
  // they correspond to fresh allocations. Otherwise, they may alias anything.
  // Even if a function is marked as not reading from memory or arguments, it
  // may still create pointers "out of the thin air", e.g., by "ptrtoint" from a
  // constant or an argument.
  auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  if (!symbol)
    return markResultsUnknown();
  auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      call, symbol.getLeafReference());
  if (!callee)
    return markResultsUnknown();

  for (OpResult result : call->getResults()) {
    AliasClassLattice *resultLattice = results[result.getResultNumber()];
    if (callee.getResultAttr(result.getResultNumber(),
                             LLVM::LLVMDialect::getNoAliasAttrName())) {
      propagateIfChanged(
          resultLattice,
          resultLattice->markFresh(call->getAttrOfType<StringAttr>("tag")));
    } else {
      propagateIfChanged(resultLattice, resultLattice->markUnknown());
    }
  }
}
