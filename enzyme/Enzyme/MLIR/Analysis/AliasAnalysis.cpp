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

const enzyme::AliasClassSet enzyme::AliasClassSet::undefinedSet =
    AliasClassSet(enzyme::AliasClassSet::State::Undefined);
const enzyme::AliasClassSet enzyme::AliasClassSet::unknownSet =
    AliasClassSet(enzyme::AliasClassSet::State::Unknown);

ChangeResult enzyme::AliasClassSet::join(const AliasClassSet &other) {
  if (isUnknown())
    return ChangeResult::NoChange;
  if (isUndefined() && other.isUndefined())
    return ChangeResult::NoChange;
  if (other.isUnknown()) {
    state = State::Unknown;
    return ChangeResult::Change;
  }

  ChangeResult result = updateStateToDefined();
  return insert(other.aliasClasses) | result;
}

ChangeResult
enzyme::AliasClassSet::insert(const DenseSet<DistinctAttr> &classes) {
  if (isUnknown())
    return ChangeResult::NoChange;

  size_t oldSize = aliasClasses.size();
  aliasClasses.insert(classes.begin(), classes.end());
  ChangeResult result = aliasClasses.size() == oldSize ? ChangeResult::NoChange
                                                       : ChangeResult::Change;
  return updateStateToDefined() | result;
}

static DistinctAttr getFreshAttr(Attribute debugLabel) {
  return DistinctAttr::create(debugLabel);
}

enzyme::AliasClassSet enzyme::AliasClassSet::getFresh(Attribute debugLabel) {
  assert(debugLabel && "expected attribute");
  AliasClassSet set(State::Defined);
  set.aliasClasses.insert(getFreshAttr(debugLabel));
  return set;
}

ChangeResult enzyme::AliasClassSet::markUnknown() {
  if (isUnknown())
    return ChangeResult::NoChange;

  state = State::Unknown;
  aliasClasses.clear();
  return ChangeResult::Change;
}

bool enzyme::AliasClassSet::isCanonical() const {
  return state == State::Defined || aliasClasses.empty();
}

bool enzyme::AliasClassSet::operator==(
    const enzyme::AliasClassSet &other) const {
  assert(isCanonical() && other.isCanonical());
  return state == other.state && llvm::equal(aliasClasses, other.aliasClasses);
}

ChangeResult enzyme::AliasClassSet::foreachClass(
    function_ref<ChangeResult(DistinctAttr, State)> callback) const {
  if (state != State::Defined)
    return callback(nullptr, state);

  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr attr : aliasClasses)
    result |= callback(attr, state);
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
  if (pointsTo.empty()) {
    llvm::errs() << "<empty>\n";
    return;
  }
  for (const auto &[srcClass, destClasses] : pointsTo) {
    os << "  " << srcClass << " points to {";
    if (destClasses.isUnknown()) {
      os << "<unknown>";
    } else if (destClasses.isUndefined()) {
      os << "<undefined>";
    } else {
      llvm::interleaveComma(destClasses.getAliasClasses(), os);
    }
    os << "}\n";
  }
  // os << "other points to unknown: " << otherPointToUnknown << "\n";
}

/// Union for every variable.
ChangeResult enzyme::PointsToSets::join(const AbstractDenseLattice &lattice) {
  const auto &rhs = static_cast<const PointsToSets &>(lattice);
  llvm::SmallDenseSet<DistinctAttr> keys;
  auto lhsRange = llvm::make_first_range(pointsTo);
  auto rhsRange = llvm::make_first_range(rhs.pointsTo);
  keys.insert(lhsRange.begin(), lhsRange.end());
  keys.insert(rhsRange.begin(), rhsRange.end());

  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr key : keys) {
    auto lhsIt = pointsTo.find(key);
    auto rhsIt = rhs.pointsTo.find(key);
    assert(lhsIt != pointsTo.end() || rhsIt != rhs.pointsTo.end());

    // If present in both, join.
    if (lhsIt != pointsTo.end() && rhsIt != rhs.pointsTo.end()) {
      result |= lhsIt->getSecond().join(rhsIt->getSecond());
      continue;
    }

    // Copy from RHS if available only there.
    if (lhsIt == pointsTo.end()) {
      pointsTo.try_emplace(rhsIt->getFirst(), rhsIt->getSecond());
      result = ChangeResult::Change;
    }

    // Do nothing if available only in LHS.
  }
  return result;
}

ChangeResult
enzyme::PointsToSets::joinPotentiallyMissing(DistinctAttr key,
                                             const AliasClassSet &value) {
  // Don't store explicitly undefined values in the mapping, keys absent from
  // the mapping are treated as implicitly undefined.
  if (value.isUndefined())
    return ChangeResult::NoChange;

  bool inserted;
  decltype(pointsTo.begin()) iterator;
  std::tie(iterator, inserted) = pointsTo.try_emplace(key, value);
  if (!inserted)
    return iterator->second.join(value);
  return ChangeResult::Change;
}

ChangeResult enzyme::PointsToSets::update(const AliasClassSet &keysToUpdate,
                                          const AliasClassSet &values,
                                          bool replace) {
  if (keysToUpdate.isUnknown())
    return markAllPointToUnknown();

  // Don't yet know what to update.
  if (keysToUpdate.isUndefined())
    return ChangeResult::NoChange;

  // Otherwise just set the result.
  // TODO(zinenko): this is still needed because a _new_ distinct attribute is
  // created every time, so we must override here. This shouldn't be necessary
  // after fixing district attribute generation to be the same for the same
  // value/operation.
  if (replace) {
    return keysToUpdate.foreachClass(
        [&](DistinctAttr dest, AliasClassSet::State state) {
          assert(state == AliasClassSet::State::Defined &&
                 "unknown must have been handled above");

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

  return keysToUpdate.foreachClass(
      [&](DistinctAttr dest, AliasClassSet::State state) {
        assert(state == AliasClassSet::State::Defined &&
               "unknown must have been handled above");
        // TODO(zinenko): enable this after removing the code above.
        // #ifndef NDEBUG
        //         if (replace) {
        //           auto it = pointsTo.find(dest);
        //           if (it != pointsTo.end()) {
        //             // Check that we are updating to a state that's >= in the
        //             lattice. AliasClassSet valuesCopy(values);
        //             valuesCopy.join(it->getSecond());
        //             values.print(llvm::errs());
        //             llvm::errs() << "\n";
        //             it->getSecond().print(llvm::errs());
        //             llvm::errs() << "\n";
        //             valuesCopy.print(llvm::errs());
        //             llvm::errs() << "\n\n";
        //             assert(valuesCopy == values &&
        //                    "attempting to replace a pointsTo entry with an
        //                    alias class " "set that is ordered _before_ the
        //                    existing one -> " "non-monotonous update ");
        //           }
        //         }
        // #endif // NDEBUG
        return joinPotentiallyMissing(dest, values);
      });
}

ChangeResult
enzyme::PointsToSets::setPointingToFresh(const AliasClassSet &destClasses,
                                         StringAttr debugLabel) {
  return update(destClasses, getFreshAttr(debugLabel),
                /*replace=*/true);
}

ChangeResult
enzyme::PointsToSets::setPointingToEmpty(const AliasClassSet &destClasses) {
  return update(destClasses, AliasClassSet::getEmpty(), /*replace=*/true);
}

ChangeResult
enzyme::PointsToSets::addSetsFrom(const AliasClassSet &destClasses,
                                  const AliasClassSet &srcClasses) {
  if (destClasses.isUnknown())
    return markAllPointToUnknown();
  if (destClasses.isUndefined())
    return ChangeResult::NoChange;

  return destClasses.foreachClass(
      [&](DistinctAttr dest, AliasClassSet::State destState) {
        assert(destState == AliasClassSet::State::Defined);
        return srcClasses.foreachClass(
            [&](DistinctAttr src, AliasClassSet::State srcState) {
              const AliasClassSet *srcClasses = &AliasClassSet::getUndefined();
              if (srcState == AliasClassSet::State::Unknown)
                srcClasses = &AliasClassSet::getUnknown();
              else if (srcState == AliasClassSet::State::Defined) {
                auto it = pointsTo.find(src);
                if (it != pointsTo.end())
                  srcClasses = &it->getSecond();
              }
              return joinPotentiallyMissing(dest, *srcClasses);
            });
      });
}

ChangeResult
enzyme::PointsToSets::markPointToUnknown(const AliasClassSet &destClasses) {
  if (destClasses.isUnknown())
    return markAllPointToUnknown();
  if (destClasses.isUndefined())
    return ChangeResult::NoChange;

  return destClasses.foreachClass([&](DistinctAttr dest, AliasClassSet::State) {
    return joinPotentiallyMissing(dest, AliasClassSet::getUnknown());
  });
}

ChangeResult enzyme::PointsToSets::markAllPointToUnknown() {
  ChangeResult result = ChangeResult::NoChange;
  for (auto &it : pointsTo)
    result |= it.getSecond().join(AliasClassSet::getUnknown());
  return result;
}

ChangeResult enzyme::PointsToSets::markAllExceptPointToUnknown(
    const AliasClassSet &destClasses) {
  if (destClasses.isUndefined())
    return ChangeResult::NoChange;

  ChangeResult result = ChangeResult::NoChange;
  for (auto &[key, value] : pointsTo) {
    if (destClasses.isUnknown() ||
        !destClasses.getAliasClasses().contains(key)) {
      result |= value.markUnknown();
    }
  }

#ifndef NDEBUG
  (void)destClasses.foreachClass(
      [&](DistinctAttr dest, AliasClassSet::State state) {
        if (state == AliasClassSet::State::Defined)
          assert(pointsTo.contains(dest) && "unknown dest cannot be preserved");
        return ChangeResult::NoChange;
      });
#endif // NDEBUG

  return result;
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

/// Returns information indicating whether the function may read or write into
/// the memory pointed to by its arguments. When unknown, returns `nullopt`.
static std::optional<LLVM::ModRefInfo>
getFunctionArgModRef(FunctionOpInterface func) {
  // First, handle some library functions with statically known behavior.
  StringRef name = cast<SymbolOpInterface>(func.getOperation()).getName();
  auto hardcoded = llvm::StringSwitch<std::optional<LLVM::ModRefInfo>>(name)
                       // printf: only reads from arguments.
                       .Case("printf", LLVM::ModRefInfo::Ref)
                       // operator delete(void *) doesn't read from arguments.
                       .Case("_ZdlPv", LLVM::ModRefInfo::NoModRef)
                       .Default(std::nullopt);
  if (hardcoded)
    return hardcoded;

  if (auto memoryAttr =
          func->getAttrOfType<LLVM::MemoryEffectsAttr>(kLLVMMemoryAttrName))
    return memoryAttr.getArgMem();
  return std::nullopt;
}

/// Returns information indicating whether the function may read or write into
/// the memory other than that pointed to by its arguments, though still
/// accessible from (any) calling context. When unknown, returns `nullopt`.
static std::optional<LLVM::ModRefInfo>
getFunctionOtherModRef(FunctionOpInterface func) {
  // First, handle some library functions with statically known behavior.
  StringRef name = cast<SymbolOpInterface>(func.getOperation()).getName();
  auto hardcoded =
      llvm::StringSwitch<std::optional<LLVM::ModRefInfo>>(name)
          // printf: doesn't access other (technically, stdout is pointer-like,
          // but we cannot flow information through it since it is write-only.
          .Case("printf", LLVM::ModRefInfo::NoModRef)
          // operator delete(void *) doesn't access other.
          .Case("_ZdlPv", LLVM::ModRefInfo::NoModRef)
          .Default(std::nullopt);
  if (hardcoded)
    return hardcoded;

  if (auto memoryAttr =
          func->getAttrOfType<LLVM::MemoryEffectsAttr>(kLLVMMemoryAttrName))
    return memoryAttr.getOther();
  return std::nullopt;
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
      // TODO(zinenko): update to using callback.
      if (memPtr->isUndefined() || memPtr->isUnknown())
        return;
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
      std::optional<LLVM::ModRefInfo> argModRef = getFunctionArgModRef(callee);
      std::optional<LLVM::ModRefInfo> otherModRef =
          getFunctionOtherModRef(callee);

      SmallVector<int> pointerLikeOperands;
      for (auto &&[i, operand] : llvm::enumerate(call.getArgOperands())) {
        if (isPointerLike(operand.getType()))
          pointerLikeOperands.push_back(i);
      }

      // Precompute the set of alias classes the function may capture.
      // TODO: consider a more advanced lattice that can encode "may point to
      // any class _except_ the given classes"; this is mathematically possible
      // but needs careful programmatic encoding.
      AliasClassSet functionMayCapture = AliasClassSet::getUndefined();
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

      // For each alias class the function may write to, indicate potentially
      // stored classes. Keep the set of writable alias classes for future.
      AliasClassSet pointerOperandClasses = AliasClassSet::getUndefined();
      ChangeResult changed = ChangeResult::NoChange;
      for (int pointerOperand : pointerLikeOperands) {
        auto *destClasses = getOrCreateFor<AliasClassLattice>(
            call, call.getArgOperands()[pointerOperand]);
        pointerOperandClasses.join(destClasses->getAliasClassesObject());

        // If the argument cannot be stored into, just preserve it as is.
        if (!mayWriteArg(callee, pointerOperand, argModRef))
          continue;

        // If the destination class is unknown, mark all known classes
        // pessimistic (alias classes that have not beed analyzed and thus are
        // absent from pointsTo are treated as "undefined" at this point).
        if (destClasses->isUnknown()) {
          pointerOperandClasses.markUnknown();
          changed |= after->markAllPointToUnknown();
          break;
        }

        if (destClasses->isUndefined())
          continue;

        // Otherwise, indicate that a pointer that belongs to any of the
        // classes captured by this function may be stored into the
        // destination class.
        changed |= destClasses->getAliasClassesObject().foreachClass(
            [&](DistinctAttr dest, AliasClassSet::State) {
              return after->insert(dest, functionMayCapture);
            });
      }

      // If the function may write to "other", that is any potential other
      // pointer, record that.
      if (modRefMayMod(otherModRef)) {
        // Classes that have been analyzed, and therefore present in the `after`
        // lattice after joining it with `before` are marked as pointing to
        // "unknown", except the classes that are associated with operands for
        // which we have more specific information. Classes that haven't been
        // analyzed, and therefore absent in the `after` lattice, are left
        // unmodified and thus assumed to be "undefined". This makes this
        // transfer function monotonic as opposed to marking the latter classes
        // as "unknown" eagerly, which would require rolling that marking back.
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

        AliasClassSet commonReturnScope = AliasClassSet::getFresh(
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
          changed |= after->insert(destClasses->getAliasClassesObject(),
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

void enzyme::AliasClassSet::print(raw_ostream &os) const {
  if (isUnknown()) {
    os << "<unknown>";
  } else if (isUndefined()) {
    os << "<undefined>";
  } else {
    llvm::interleaveComma(aliasClasses, os << "{");
    os << "}";
  }
}

void enzyme::AliasClassLattice::print(raw_ostream &os) const {
  if (aliasClasses.isUnknown()) {
    os << "Unknown AC";
  } else if (aliasClasses.isUndefined()) {
    os << "Undefined AC";
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
  const auto *otherAliasClass = static_cast<const AliasClassLattice *>(&other);
  return aliasClasses.join(otherAliasClass->aliasClasses);
}

enzyme::AliasClassLattice
enzyme::AliasClassLattice::getFresh(Value point, Attribute debugLabel) {
  if (!debugLabel)
    debugLabel = UnitAttr::get(point.getContext());
  return AliasClassLattice(point, AliasClassSet::getFresh(debugLabel));
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
        // TODO: this may currently be failing because `setToEntryState`
        // is used by the framework to set the pessimistic fixpoint (top), which
        // isn't correct for pessimistic analysis for which `setToEntryState` is
        // the undefined state (bottom).
        assert(lattice->isUndefined() && "resetting lattice point");

        return propagateIfChanged(lattice,
                                  lattice->join(AliasClassLattice::getFresh(
                                      lattice->getPoint(), debugLabel)));
      }
      // TODO: Not safe in general, integers can be a result of ptrtoint. We
      // need a type analysis here I guess?
      if (isPointerLike(arg.getType()))
        return propagateIfChanged(lattice, lattice->insert({entryClass}));
    }
  }
  if (!lattice->isUndefined())
    llvm::errs() << *lattice << "\n";
  assert(lattice->isUndefined());
  // The default state is "undefined", no need to explicitly (re)set it.
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
      // Mark the result of the allocation as a fresh memory location.
      for (AliasClassLattice *result : results) {
        if (result->getPoint() == value) {
          Attribute debugLabel = op->getAttr("tag");
          // TODO(zinenko): here and other cases of `markFresh` are dangerous,
          // if these are executed repeatedly, they will be associated with a
          // _new_ fresh distinct attribute every time. This has always been
          // the case with `markFresh`, we need to store a unique mapping
          // somewhere.
          auto fresh =
              AliasClassLattice::getFresh(result->getPoint(), debugLabel);
          propagateIfChanged(result, result->join(fresh));

          // The pointer to freshly allocated memory is known not to point to
          // anything.
          // TODO(zinenko): this is a bit strange to update _another_ lattice
          // here.
          auto *pointsTo = getOrCreate<PointsToSets>(op);
          propagateIfChanged(pointsTo, pointsTo->setPointingToEmpty(
                                           fresh.getAliasClassesObject()));
        }
      }
    } else if (isa<MemoryEffects::Read>(effect.getEffect())) {
      auto *pointsToSets = getOrCreateFor<PointsToSets>(op, op);
      AliasClassLattice *latticeElement = getLatticeElement(value);
      if (latticeElement->isUnknown()) {
        for (AliasClassLattice *result : results) {
          propagateIfChanged(result, result->markUnknown());
        }
      } else if (latticeElement->isUndefined()) {
        // Do nothing unless we know something about the value.
      } else {
        for (auto srcClass : latticeElement->getAliasClasses()) {
          const auto &srcPointsTo = pointsToSets->getPointsTo(srcClass);
          for (AliasClassLattice *result : results) {
            // TODO: consider some sort of "point join" or better insert that
            // doesn't require a conditional here.
            if (srcPointsTo.isUnknown()) {
              propagateIfChanged(result, result->markUnknown());
            } else if (srcPointsTo.isUndefined()) {
              continue;
            } else {
              propagateIfChanged(result,
                                 result->insert(srcPointsTo.getAliasClasses()));
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
      Attribute debugLabel = call->getAttrOfType<StringAttr>("tag");
      propagateIfChanged(resultLattice,
                         resultLattice->join(AliasClassLattice::getFresh(
                             resultLattice->getPoint(), debugLabel)));
    } else {
      propagateIfChanged(resultLattice, resultLattice->markUnknown());
    }
  }
}
