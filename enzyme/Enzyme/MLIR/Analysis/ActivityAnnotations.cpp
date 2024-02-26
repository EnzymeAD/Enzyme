#include "ActivityAnnotations.h"
#include "AliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Lattice.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/raw_ostream.h"

// TODO: Remove dependency on dialects in favour of differential dependency
// interface
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

static StringRef getActivityAnnotationAttrName() { return "activedeps"; }
static StringRef getPointerSummaryAttrName() { return "p2psummary"; }
static StringRef getReturnOriginsAttrName() { return "returnorigins"; }

template <typename ValueT>
void printSetLattice(const enzyme::SparseSetLattice<ValueT> &setLattice,
                     raw_ostream &os) {
  if (setLattice.isUnknown()) {
    os << "Unknown Origin";
  } else if (setLattice.isUndefined()) {
    os << "Undefined Origin";
  } else {
    os << "size: " << setLattice.getElements().size() << ":\n";
    for (auto element : setLattice.getElements()) {
      os << "  " << element << "\n";
    }
  }
}

void enzyme::ForwardOriginsLattice::print(raw_ostream &os) const {
  printSetLattice(*this, os);
}

void enzyme::BackwardOriginsLattice::print(raw_ostream &os) const {
  printSetLattice(*this, os);
}

ChangeResult
enzyme::ForwardOriginsLattice::join(const AbstractSparseLattice &other) {
  const auto *otherValueOrigins =
      static_cast<const ForwardOriginsLattice *>(&other);
  return elements.join(otherValueOrigins->elements);
}

void enzyme::ForwardActivityAnnotationAnalysis::setToEntryState(
    ForwardOriginsLattice *lattice) {
  auto arg = dyn_cast<BlockArgument>(lattice->getPoint());
  if (!arg) {
    assert(lattice->isUndefined());
    return;
  }

  auto funcOp = cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
  auto origin = ArgumentOriginAttr::get(FlatSymbolRefAttr::get(funcOp),
                                        arg.getArgNumber());
  return propagateIfChanged(
      lattice, lattice->join(
                   ForwardOriginsLattice::single(lattice->getPoint(), origin)));
}

/// True iff all results differentially depend on all operands
// TODO: differential dependency/activity interface
static bool isFullyActive(Operation *op) {
  return isa<LLVM::FMulOp, LLVM::FAddOp, LLVM::FDivOp, LLVM::FSubOp,
             LLVM::FNegOp, LLVM::FAbsOp, LLVM::SqrtOp, LLVM::SinOp, LLVM::CosOp,
             LLVM::Exp2Op, LLVM::ExpOp, LLVM::InsertValueOp,
             LLVM::ExtractValueOp, LLVM::BitcastOp>(op);
}

static bool isNoOp(Operation *op) {
  return isa<LLVM::NoAliasScopeDeclOp, LLVM::LifetimeStartOp,
             LLVM::LifetimeEndOp>(op);
}

void enzyme::ForwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, ArrayRef<const ForwardOriginsLattice *> operands,
    ArrayRef<ForwardOriginsLattice *> results) {
  if (isFullyActive(op)) {
    for (ForwardOriginsLattice *result : results) {
      for (const ForwardOriginsLattice *operand : operands) {
        join(result, *operand);
      }
    }
    return;
  }

  // Expected to be handled through the diff dependency interface
  if (isPure(op) || isNoOp(op))
    return;

  auto markResultsUnknown = [&]() {
    for (ForwardOriginsLattice *result : results) {
      propagateIfChanged(result, result->markUnknown());
    }
  };

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    return markResultsUnknown();
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Read>(effect.getEffect()))
      continue;

    Value value = effect.getValue();
    if (!value) {
      markResultsUnknown();
      continue;
    }
    processMemoryRead(op, value, results);
  }
}

void enzyme::ForwardActivityAnnotationAnalysis::processMemoryRead(
    Operation *op, Value address, ArrayRef<ForwardOriginsLattice *> results) {
  auto markResultsUnknown = [&]() {
    for (ForwardOriginsLattice *result : results) {
      propagateIfChanged(result, result->markUnknown());
    }
  };

  auto *srcClasses = getOrCreateFor<AliasClassLattice>(op, address);
  auto *originsMap = getOrCreateFor<ForwardOriginsMap>(op, op);
  if (srcClasses->isUndefined())
    return;
  if (srcClasses->isUnknown())
    return markResultsUnknown();

  // Look up the alias class and see what its origins are, then propagate
  // those origins to the read results.
  for (DistinctAttr srcClass : srcClasses->getAliasClasses()) {
    for (ForwardOriginsLattice *result : results) {
      propagateIfChanged(result,
                         result->merge(originsMap->getOrigins(srcClass)));
    }
  }
}

void deserializeReturnOrigins(ArrayAttr returnOrigins,
                              SmallVectorImpl<enzyme::ValueOriginSet> &out) {
  for (auto &&[resultIdx, argOrigins] : llvm::enumerate(returnOrigins)) {
    enzyme::ValueOriginSet origins;
    if (auto strAttr = dyn_cast<StringAttr>(argOrigins)) {
      if (strAttr.getValue() == "<unknown>") {
        (void)origins.markUnknown();
      } else {
        // Leave origins undefined
      }
    } else {
      for (enzyme::ArgumentOriginAttr originAttr :
           cast<ArrayAttr>(argOrigins)
               .getAsRange<enzyme::ArgumentOriginAttr>()) {
        (void)origins.insert({originAttr});
      }
    }

    out.push_back(origins);
  }
}

void enzyme::ForwardActivityAnnotationAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const ForwardOriginsLattice *> operands,
    ArrayRef<ForwardOriginsLattice *> results) {
  auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  auto markAllResultsUnknown = [&]() {
    for (ForwardOriginsLattice *result : results) {
      propagateIfChanged(result, result->markUnknown());
    }
  };
  if (!symbol)
    return markAllResultsUnknown();

  if (auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
          call, symbol.getLeafReference())) {
    if (auto returnOriginsAttr =
            callee->getAttrOfType<ArrayAttr>(getReturnOriginsAttrName())) {
      SmallVector<ValueOriginSet> returnOrigins;
      deserializeReturnOrigins(returnOriginsAttr, returnOrigins);
      return processCallToSummarizedFunc(call, returnOrigins, operands,
                                         results);
    }
  }
}

void enzyme::ForwardActivityAnnotationAnalysis::processCallToSummarizedFunc(
    CallOpInterface call, ArrayRef<ValueOriginSet> summary,
    ArrayRef<const ForwardOriginsLattice *> operands,
    ArrayRef<ForwardOriginsLattice *> results) {
  for (const auto &[result, returnOrigin] : llvm::zip(results, summary)) {
    // Convert the origins relative to the callee to relative to the caller
    ValueOriginSet callerOrigins;
    if (returnOrigin.isUndefined())
      continue;

    if (returnOrigin.isUnknown()) {
      (void)callerOrigins.markUnknown();
    } else {
      (void)returnOrigin.foreachElement(
          [&](OriginAttr calleeOrigin, ValueOriginSet::State state) {
            assert(state == ValueOriginSet::State::Defined &&
                   "undefined and unknown must have been handled above");
            auto calleeArgOrigin = cast<ArgumentOriginAttr>(calleeOrigin);
            return callerOrigins.join(
                operands[calleeArgOrigin.getArgNumber()]->getOriginsObject());
          });
    }
    propagateIfChanged(result, result->merge(callerOrigins));
  }
}

void enzyme::BackwardActivityAnnotationAnalysis::setToExitState(
    BackwardOriginsLattice *lattice) {
  propagateIfChanged(lattice, lattice->markUnknown());
}

void enzyme::BackwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, ArrayRef<BackwardOriginsLattice *> operands,
    ArrayRef<const BackwardOriginsLattice *> results) {
  if (isFullyActive(op)) {
    for (BackwardOriginsLattice *operand : operands)
      for (const BackwardOriginsLattice *result : results)
        meet(operand, *result);
  }

  // Expected to be handled through the diff dependency interface
  if (isPure(op) || isNoOp(op))
    return;

  auto markOperandsUnknown = [&]() {
    for (BackwardOriginsLattice *operand : operands) {
      propagateIfChanged(operand, operand->markUnknown());
    }
  };

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    return markOperandsUnknown();
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Read>(effect.getEffect()))
      continue;

    Value value = effect.getValue();
    if (!value) {
      markOperandsUnknown();
      continue;
    }

    auto *srcClasses = getOrCreateFor<AliasClassLattice>(op, value);
    auto *originsMap = getOrCreate<BackwardOriginsMap>(op);

    ChangeResult changed = ChangeResult::NoChange;
    for (const BackwardOriginsLattice *result : results)
      changed |= originsMap->insert(srcClasses->getAliasClassesObject(),
                                    result->getOriginsObject());
    propagateIfChanged(originsMap, changed);
  }
}

void enzyme::BackwardActivityAnnotationAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<BackwardOriginsLattice *> operands,
    ArrayRef<const BackwardOriginsLattice *> results) {
  auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  auto markAllOperandsUnknown = [&]() {
    for (BackwardOriginsLattice *operand : operands) {
      propagateIfChanged(operand, operand->markUnknown());
    }
  };
  if (!symbol)
    return markAllOperandsUnknown();

  if (auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
          call, symbol.getLeafReference())) {
    if (auto returnOriginsAttr =
            callee->getAttrOfType<ArrayAttr>(getReturnOriginsAttrName())) {
      SmallVector<ValueOriginSet> returnOrigins;
      deserializeReturnOrigins(returnOriginsAttr, returnOrigins);
      return processCallToSummarizedFunc(call, returnOrigins, operands,
                                         results);
    }
  }
}

void enzyme::BackwardActivityAnnotationAnalysis::processCallToSummarizedFunc(
    CallOpInterface call, ArrayRef<ValueOriginSet> summary,
    ArrayRef<BackwardOriginsLattice *> operands,
    ArrayRef<const BackwardOriginsLattice *> results) {
  // collect the result origins, propagate them to the operands.
  for (const auto &[result, calleeOrigins] : llvm::zip(results, summary)) {
    ValueOriginSet resultOrigins = result->getOriginsObject();
    if (calleeOrigins.isUndefined())
      continue;
    if (calleeOrigins.isUnknown())
      (void)resultOrigins.markUnknown();
    else {
      (void)calleeOrigins.foreachElement(
          [&](OriginAttr calleeOrigin, ValueOriginSet::State state) {
            auto calleeArgOrigin = cast<ArgumentOriginAttr>(calleeOrigin);
            BackwardOriginsLattice *operand =
                operands[calleeArgOrigin.getArgNumber()];
            propagateIfChanged(operand, operand->merge(resultOrigins));
            return ChangeResult::NoChange;
          });
    }
  }
}

template <typename KeyT, typename ElementT>
void printMapOfSetsLattice(
    const DenseMap<KeyT, enzyme::SetLattice<ElementT>> map, raw_ostream &os) {
  if (map.empty()) {
    os << "<empty>\n";
    return;
  }
  for (const auto &[aliasClass, origins] : map) {
    os << "  " << aliasClass << " originates from {";
    if (origins.isUnknown()) {
      os << "<unknown>";
    } else if (origins.isUndefined()) {
      os << "<undefined>";
    } else {
      llvm::interleaveComma(origins.getElements(), os);
    }
    os << "}\n";
  }
}

void enzyme::ForwardOriginsMap::print(raw_ostream &os) const {
  printMapOfSetsLattice(this->map, os);
}

void enzyme::BackwardOriginsMap::print(raw_ostream &os) const {
  printMapOfSetsLattice(this->map, os);
}

bool enzyme::sortAttributes(Attribute a, Attribute b) {
  auto originA = dyn_cast<enzyme::ArgumentOriginAttr>(a);
  auto originB = dyn_cast<enzyme::ArgumentOriginAttr>(b);
  if (originA && originB)
    return originA.getArgNumber() < originB.getArgNumber();

  auto distinctA = dyn_cast<DistinctAttr>(a);
  auto distinctB = dyn_cast<DistinctAttr>(b);
  // If not distinct attributes, sort them arbitrarily.
  if (!(distinctA && distinctB))
    return &a < &b;

  auto pseudoA = dyn_cast_if_present<enzyme::PseudoAliasClassAttr>(
      distinctA.getReferencedAttr());
  auto pseudoB = dyn_cast_if_present<enzyme::PseudoAliasClassAttr>(
      distinctB.getReferencedAttr());
  auto strA = dyn_cast_if_present<StringAttr>(distinctA.getReferencedAttr());
  auto strB = dyn_cast_if_present<StringAttr>(distinctB.getReferencedAttr());
  if (pseudoA && pseudoB) {
    return std::make_pair(pseudoA.getArgNumber(), pseudoA.getDepth()) <
           std::make_pair(pseudoB.getArgNumber(), pseudoB.getDepth());
  } else if (strA && strB) {
    return strA.strref() < strB.strref();
  }
  // Order pseudo/origin classes before fresh classes
  return (pseudoA || originA) && !(pseudoB || originB);
}

void enzyme::DenseActivityAnnotationAnalysis::setToEntryState(
    ForwardOriginsMap *lattice) {
  auto *block = dyn_cast<Block *>(lattice->getPoint());
  if (!block)
    return;

  auto funcOp = cast<FunctionOpInterface>(block->getParentOp());
  ChangeResult changed = ChangeResult::NoChange;
  for (BlockArgument arg : funcOp.getArguments()) {
    auto *argClass = getOrCreateFor<AliasClassLattice>(block, arg);
    auto origin = ArgumentOriginAttr::get(FlatSymbolRefAttr::get(funcOp),
                                          arg.getArgNumber());
    changed |= lattice->insert(argClass->getAliasClassesObject(),
                               ValueOriginSet(origin));
  }
  propagateIfChanged(lattice, changed);
}

std::optional<Value> getStored(Operation *op);
std::optional<Value> getCopySource(Operation *op);

void enzyme::DenseActivityAnnotationAnalysis::visitOperation(
    Operation *op, const ForwardOriginsMap &before, ForwardOriginsMap *after) {
  join(after, before);

  if (isNoOp(op))
    return;

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    return propagateIfChanged(after, after->markAllOriginsUnknown());
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    if (!value)
      return propagateIfChanged(after, after->markAllOriginsUnknown());

    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      // TODO: Really need that memory interface
      if (op->getNumResults() != 1)
        continue;
      Value readDest = op->getResult(0);

      auto *destClasses = getOrCreateFor<AliasClassLattice>(op, readDest);
      if (destClasses->isUndefined())
        // Not a pointer, so the sparse analysis will handle this.
        continue;

      auto *srcClasses = getOrCreateFor<AliasClassLattice>(op, value);
      if (srcClasses->isUnknown()) {
        propagateIfChanged(after,
                           after->insert(destClasses->getAliasClassesObject(),
                                         ValueOriginSet::getUnknown()));
        continue;
      }

      ChangeResult changed = ChangeResult::NoChange;
      for (DistinctAttr srcClass : srcClasses->getAliasClasses()) {
        changed |= after->insert(destClasses->getAliasClassesObject(),
                                 before.getOrigins(srcClass));
      }
      propagateIfChanged(after, changed);
    } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
      if (std::optional<Value> stored = getStored(op)) {
        auto *origins = getOrCreateFor<ForwardOriginsLattice>(op, *stored);
        auto *dest = getOrCreateFor<AliasClassLattice>(op, value);
        propagateIfChanged(after, after->insert(dest->getAliasClassesObject(),
                                                origins->getOriginsObject()));
      } else if (std::optional<Value> copySource = getCopySource(op)) {
        processCopy(op, *copySource, value, before, after);
      } else {
        propagateIfChanged(after, after->markAllOriginsUnknown());
      }
    }
  }
}

void enzyme::DenseActivityAnnotationAnalysis::processCopy(
    Operation *op, Value copySource, Value copyDest,
    const ForwardOriginsMap &before, ForwardOriginsMap *after) {
  auto *src = getOrCreateFor<AliasClassLattice>(op, copySource);
  ValueOriginSet srcOrigins;
  if (src->isUndefined())
    return;
  if (src->isUnknown())
    (void)srcOrigins.markUnknown();

  for (DistinctAttr srcClass : src->getAliasClasses())
    (void)srcOrigins.join(before.getOrigins(srcClass));

  auto *dest = getOrCreateFor<AliasClassLattice>(op, copyDest);
  propagateIfChanged(after,
                     after->insert(dest->getAliasClassesObject(), srcOrigins));
}

// TODO: rename from pointsto
static void deserializePointsTo(
    ArrayAttr summaryAttr,
    DenseMap<DistinctAttr, enzyme::ValueOriginSet> &summaryMap) {
  // TODO: investigate better encodings for the value origin summary
  for (auto pair : summaryAttr.getAsRange<ArrayAttr>()) {
    assert(pair.size() == 2 &&
           "Expected summary to be in [[key, value]] format");
    auto pointer = cast<DistinctAttr>(pair[0]);
    auto pointsToSet = enzyme::ValueOriginSet::getUndefined();
    if (auto strAttr = dyn_cast<StringAttr>(pair[1])) {
      if (strAttr.getValue() == "unknown") {
        (void)pointsToSet.markUnknown();
      } else {
        assert(strAttr.getValue() == "undefined" &&
               "unrecognized points-to destination");
      }
    } else {
      auto pointsTo = cast<ArrayAttr>(pair[1]).getAsRange<enzyme::OriginAttr>();
      (void)pointsToSet.insert(
          DenseSet<enzyme::OriginAttr>(pointsTo.begin(), pointsTo.end()));
    }

    summaryMap.insert({pointer, pointsToSet});
  }
}

void enzyme::DenseActivityAnnotationAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const ForwardOriginsMap &before, ForwardOriginsMap *after) {
  join(after, before);
  if (action == dataflow::CallControlFlowAction::ExternalCallee) {
    auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
    if (!symbol)
      return propagateIfChanged(after, after->markAllOriginsUnknown());

    if (auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
            call, symbol.getLeafReference())) {
      if (auto summaryAttr = callee->getAttrOfType<ArrayAttr>(
              getActivityAnnotationAttrName())) {
        DenseMap<DistinctAttr, ValueOriginSet> summary;
        deserializePointsTo(summaryAttr, summary);
        return processCallToSummarizedFunc(call, summary, before, after);
      }
    }
  }
}

/// Visit everything transitively pointed-to by any pointer in start.
static void traversePointsToSets(const enzyme::AliasClassSet &start,
                                 const enzyme::PointsToSets &pointsToSets,
                                 function_ref<void(DistinctAttr)> visit) {
  using enzyme::AliasClassSet;
  AliasClassSet current(start);
  while (!current.isUndefined()) {
    AliasClassSet next;

    assert(!current.isUnknown() && "Unhandled traversal of unknown");
    for (DistinctAttr currentClass : current.getElements()) {
      visit(currentClass);
      (void)next.join(pointsToSets.getPointsTo(currentClass));
    }
    std::swap(current, next);
  }
}

void enzyme::DenseActivityAnnotationAnalysis::processCallToSummarizedFunc(
    CallOpInterface call, const DenseMap<DistinctAttr, ValueOriginSet> &summary,
    const ForwardOriginsMap &before, ForwardOriginsMap *after) {
  // StringRef calleeName = cast<SymbolRefAttr>(call.getCallableForCallee())
  //                            .getLeafReference()
  //                            .getValue();

  ChangeResult changed = ChangeResult::NoChange;
  // Unify the value origin summary with the actual lattices of function
  // arguments
  // Collect the origins of the function arguments, then collect the alias
  // classes of the destinations
  auto *p2sets = getOrCreateFor<PointsToSets>(call, call);
  SmallVector<ValueOriginSet> argumentOrigins;
  SmallVector<AliasClassSet> argumentClasses;
  for (auto &&[i, argOperand] : llvm::enumerate(call.getArgOperands())) {
    // Value origin might be sparse, might be dense
    ValueOriginSet argOrigins;
    auto *argClasses = getOrCreateFor<AliasClassLattice>(call, argOperand);
    if (argClasses->isUndefined()) {
      // Not a pointer, use the sparse lattice state
      auto *sparseOrigins =
          getOrCreateFor<ForwardOriginsLattice>(call, argOperand);
      (void)argOrigins.join(sparseOrigins->getOriginsObject());
    } else {
      // Unify all the origins
      // Since we're not keeping track of argument depth, we need to union the
      // arg origins with everything it points to.
      traversePointsToSets(argClasses->getAliasClassesObject(), *p2sets,
                           [&](DistinctAttr aliasClass) {
                             (void)argOrigins.join(
                                 before.getOrigins(aliasClass));
                           });
    }
    argumentClasses.push_back(argClasses->getAliasClassesObject());
    argumentOrigins.push_back(argOrigins);
  }

  // TODO: Does the traversal order matter here?
  for (const auto &[destClass, sourceOrigins] : summary) {
    ValueOriginSet callerOrigins;
    for (Attribute sourceOrigin : sourceOrigins.getElements()) {
      unsigned argNumber =
          cast<ArgumentOriginAttr>(sourceOrigin).getArgNumber();
      (void)callerOrigins.join(argumentOrigins[argNumber]);
    }

    AliasClassSet callerDestClasses;
    if (auto pseudoClass = dyn_cast_if_present<PseudoAliasClassAttr>(
            destClass.getReferencedAttr())) {
      // Traverse the points-to sets.
      AliasClassSet current = argumentClasses[pseudoClass.getArgNumber()];
      unsigned depth = pseudoClass.getDepth();
      while (depth > 0) {
        AliasClassSet next;
        if (current.isUndefined()) {
          // TODO: what should be done here?
          // Activity annotations requires converged pointer info. If we have
          // incomplete points-to info, we can't currently tell if it's because
          // the points-to info hasn't _yet_ been computed (in which case we
          // bail out here expecting to be called again with more complete
          // points-to info), or if the points-to info has converged, this
          // signifies reading from uninitialized memory.
          return;
        }
        for (DistinctAttr currentClass : current.getElements())
          (void)next.join(p2sets->getPointsTo(currentClass));
        std::swap(current, next);
        depth--;
      }

      (void)callerDestClasses.join(current);
    } else {
      (void)callerDestClasses.insert({destClass});
    }
    changed |= after->insert(callerDestClasses, callerOrigins);
  }
  propagateIfChanged(after, changed);
}

void enzyme::DenseBackwardActivityAnnotationAnalysis::
    visitCallControlFlowTransfer(CallOpInterface call,
                                 dataflow::CallControlFlowAction action,
                                 const BackwardOriginsMap &after,
                                 BackwardOriginsMap *before) {
  meet(before, after);
  if (action == dataflow::CallControlFlowAction::ExternalCallee) {
    auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
    if (!symbol)
      return propagateIfChanged(before, before->markAllOriginsUnknown());

    if (auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
            call, symbol.getLeafReference())) {
      if (auto summaryAttr = callee->getAttrOfType<ArrayAttr>(
              getActivityAnnotationAttrName())) {
        DenseMap<DistinctAttr, ValueOriginSet> summary;
        deserializePointsTo(summaryAttr, summary);
        return processCallToSummarizedFunc(call, summary, after, before);
      }
    }
  }
}

void enzyme::DenseBackwardActivityAnnotationAnalysis::setToExitState(
    BackwardOriginsMap *lattice) {
  auto *block = dyn_cast<Block *>(lattice->getPoint());
  if (!block)
    return;

  auto funcOp = cast<FunctionOpInterface>(block->getParentOp());
  ChangeResult changed = ChangeResult::NoChange;
  for (BlockArgument arg : funcOp.getArguments()) {
    auto *pointsToSets =
        getOrCreateFor<PointsToSets>(block, block->getTerminator());
    auto *argClass = getOrCreateFor<AliasClassLattice>(block, arg);
    auto origin = ArgumentOriginAttr::get(FlatSymbolRefAttr::get(funcOp),
                                          arg.getArgNumber());

    // Everything that a pointer argument may point to originates from that
    // pointer argument.
    traversePointsToSets(argClass->getAliasClassesObject(), *pointsToSets,
                         [&](DistinctAttr currentClass) {
                           changed |=
                               lattice->insert(AliasClassSet(currentClass),
                                               ValueOriginSet(origin));
                         });
  }
  propagateIfChanged(lattice, changed);
}

void enzyme::DenseBackwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, const BackwardOriginsMap &after,
    BackwardOriginsMap *before) {
  meet(before, after);

  if (isNoOp(op))
    return;

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    return propagateIfChanged(before, before->markAllOriginsUnknown());
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;

    Value value = effect.getValue();
    if (!value)
      return propagateIfChanged(before, before->markAllOriginsUnknown());

    if (std::optional<Value> stored = getStored(op)) {
      auto *addressClasses = getOrCreateFor<AliasClassLattice>(op, value);
      auto *storedClasses = getOrCreateFor<AliasClassLattice>(op, *stored);

      if (storedClasses->isUndefined()) {
        // Not a pointer being stored, do a sparse update
        auto *storedOrigins = getOrCreate<BackwardOriginsLattice>(*stored);
        propagateIfChanged(
            storedOrigins,
            addressClasses->getAliasClassesObject().foreachElement(
                [&](DistinctAttr alloc, AliasClassSet::State state) {
                  if (state == AliasClassSet::State::Undefined) {
                    return ChangeResult::NoChange;
                  }
                  if (state == AliasClassSet::State::Unknown) {
                    return storedOrigins->markUnknown();
                  }
                  return storedOrigins->merge(after.getOrigins(alloc));
                }));
      } else if (storedClasses->isUnknown()) {
        propagateIfChanged(before, before->markAllOriginsUnknown());
      } else {
        // Capturing stores are handled via the points-to relationship in
        // setToExitState.
      }
    } else if (std::optional<Value> copySource = getCopySource(op)) {
      processCopy(op, *copySource, value, after, before);
    }
  }
}

void enzyme::DenseBackwardActivityAnnotationAnalysis::
    processCallToSummarizedFunc(
        CallOpInterface call,
        const DenseMap<DistinctAttr, ValueOriginSet> &summary,
        const BackwardOriginsMap &after, BackwardOriginsMap *before) {
  ChangeResult changed = ChangeResult::NoChange;
  // Unify the value origin summary with the actual lattices of function
  // arguments
  auto *p2sets = getOrCreateFor<PointsToSets>(call, call);
  SmallVector<AliasClassSet> argumentClasses;
  for (Value argOperand : call.getArgOperands()) {
    auto *argClasses = getOrCreateFor<AliasClassLattice>(call, argOperand);
    argumentClasses.push_back(argClasses->getAliasClassesObject());
  }

  for (const auto &[destClass, sourceOrigins] : summary) {
    // Get the destination origins
    ValueOriginSet destOrigins;
    if (auto pseudoClass = dyn_cast_if_present<PseudoAliasClassAttr>(
            destClass.getReferencedAttr())) {
      traversePointsToSets(argumentClasses[pseudoClass.getArgNumber()], *p2sets,
                           [&](DistinctAttr aliasClass) {
                             (void)destOrigins.join(
                                 after.getOrigins(aliasClass));
                           });
    }

    if (destOrigins.isUndefined())
      continue;

    // Get the source alias classes
    AliasClassSet callerSourceClasses;
    for (Attribute sourceOrigin : sourceOrigins.getElements()) {
      unsigned argNumber =
          cast<ArgumentOriginAttr>(sourceOrigin).getArgNumber();

      if (argumentClasses[argNumber].isUndefined()) {
        // Not a pointer, do a sparse update
        raw_ostream &os = llvm::outs();
        os << "sparse update dest origins: ";
        destOrigins.print(os);
        os << "\n";
        auto *backwardLattice = getOrCreate<BackwardOriginsLattice>(
            call.getArgOperands()[argNumber]);
        if (destOrigins.isUnknown()) {
          propagateIfChanged(backwardLattice, backwardLattice->markUnknown());
          continue;
        }
        propagateIfChanged(backwardLattice,
                           backwardLattice->insert(destOrigins.getElements()));
      } else {
        traversePointsToSets(argumentClasses[argNumber], *p2sets,
                             [&](DistinctAttr aliasClass) {
                               (void)callerSourceClasses.insert({aliasClass});
                             });
      }
    }
    changed |= before->insert(callerSourceClasses, destOrigins);
  }
  propagateIfChanged(before, changed);
}

void enzyme::DenseBackwardActivityAnnotationAnalysis::processCopy(
    Operation *op, Value copySource, Value copyDest,
    const BackwardOriginsMap &after, BackwardOriginsMap *before) {
  auto *dest = getOrCreateFor<AliasClassLattice>(op, copyDest);
  ValueOriginSet destOrigins;
  if (dest->isUndefined())
    return;
  if (dest->isUnknown())
    (void)destOrigins.markUnknown();

  for (DistinctAttr destClass : dest->getAliasClasses())
    (void)destOrigins.join(after.getOrigins(destClass));

  auto *src = getOrCreateFor<AliasClassLattice>(op, copySource);
  propagateIfChanged(before,
                     before->insert(src->getAliasClassesObject(), destOrigins));
}

namespace {
/// Starting from callee, compute a reverse (bottom-up) topological sorting of
/// all functions transitively called from callee.
void reverseToposortCallgraph(CallableOpInterface callee,
                              SymbolTableCollection *symbolTable,
                              SmallVectorImpl<CallableOpInterface> &sorted) {
  DenseSet<CallableOpInterface> permanent;
  DenseSet<CallableOpInterface> temporary;
  std::function<void(CallableOpInterface)> visit =
      [&](CallableOpInterface node) {
        if (permanent.contains(node))
          return;
        if (temporary.contains(node))
          assert(false && "unimplemented cycle in call graph");

        temporary.insert(node);
        node.walk([&](CallOpInterface call) {
          auto neighbour =
              cast<CallableOpInterface>(call.resolveCallable(symbolTable));
          visit(neighbour);
        });

        temporary.erase(node);
        permanent.insert(node);
        sorted.push_back(node);
      };

  visit(callee);
}

void initializeSparseBackwardActivityAnnotations(FunctionOpInterface func,
                                                 DataFlowSolver &solver) {
  using namespace mlir::enzyme;

  for (Operation &op : func.getCallableRegion()->getOps()) {
    if (!op.hasTrait<OpTrait::ReturnLike>())
      continue;

    for (OpOperand &returnOperand : op.getOpOperands()) {
      auto *lattice =
          solver.getOrCreateState<BackwardOriginsLattice>(returnOperand.get());
      auto origin = ReturnOriginAttr::get(FlatSymbolRefAttr::get(func),
                                          returnOperand.getOperandNumber());
      (void)lattice->insert({origin});
    }
  }
}
} // namespace

void enzyme::runActivityAnnotations(FunctionOpInterface callee) {
  SymbolTableCollection symbolTable;
  SmallVector<CallableOpInterface> sorted;
  reverseToposortCallgraph(callee, &symbolTable, sorted);
  raw_ostream &os = llvm::outs();

  for (CallableOpInterface node : sorted) {
    if (!node.getCallableRegion() || node->hasAttr(getPointerSummaryAttrName()))
      continue;
    auto funcOp = cast<FunctionOpInterface>(node.getOperation());
    os << "[ata] processing function @" << funcOp.getName() << "\n";
    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);
    SymbolTableCollection symbolTable;

    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<enzyme::AliasAnalysis>(callee.getContext(),
                                       /*relative=*/true);
    solver.load<enzyme::PointsToPointerAnalysis>();
    solver.load<enzyme::ForwardActivityAnnotationAnalysis>();
    solver.load<enzyme::DenseActivityAnnotationAnalysis>();
    solver.load<enzyme::BackwardActivityAnnotationAnalysis>(symbolTable);
    solver.load<enzyme::DenseBackwardActivityAnnotationAnalysis>(symbolTable);

    initializeSparseBackwardActivityAnnotations(funcOp, solver);

    if (failed(solver.initializeAndRun(node))) {
      assert(false && "dataflow solver failed");
    }

    // Create the overall summary by joining sets at all return sites.
    enzyme::PointsToSets p2sets(nullptr);
    enzyme::ForwardOriginsMap voMap(nullptr);
    size_t numResults = node.getResultTypes().size();
    SmallVector<enzyme::ForwardOriginsLattice> returnOperandOrigins(
        numResults, ForwardOriginsLattice(nullptr));
    SmallVector<enzyme::AliasClassLattice> returnAliasClasses(
        numResults, AliasClassLattice(nullptr));

    for (Operation &op : node.getCallableRegion()->getOps()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) {
        (void)p2sets.join(*solver.lookupState<enzyme::PointsToSets>(&op));
        auto *returnOrigins =
            solver.lookupState<enzyme::ForwardOriginsMap>(&op);
        if (returnOrigins)
          (void)voMap.join(*returnOrigins);

        for (OpOperand &operand : op.getOpOperands()) {
          (void)returnAliasClasses[operand.getOperandNumber()].join(
              *solver.lookupState<enzyme::AliasClassLattice>(operand.get()));
          (void)returnOperandOrigins[operand.getOperandNumber()].join(
              *solver.lookupState<enzyme::ForwardOriginsLattice>(
                  operand.get()));
        }
      }
    }

    // for (BlockArgument arg : node.getCallableRegion()->getArguments()) {
    //   auto *backwardState =
    //       solver.getOrCreateState<enzyme::BackwardOriginsLattice>(arg);
    //   os << "[debug] backward state for arg " << arg.getArgNumber() << ": "
    //      << *backwardState << "\n";
    // }

    for (auto lattice : returnAliasClasses) {
      os << "[debug] return alias class: " << lattice << "\n";
    }

    node->setAttr(getPointerSummaryAttrName(),
                  p2sets.serialize(node.getContext()));
    os << "[ata] p2p summary:\n";
    if (node->getAttrOfType<ArrayAttr>(getPointerSummaryAttrName()).size() ==
        0) {
      os << "     <empty>\n";
    }
    for (ArrayAttr pair :
         node->getAttrOfType<ArrayAttr>(getPointerSummaryAttrName())
             .getAsRange<ArrayAttr>()) {
      os << "     " << pair[0] << " -> " << pair[1] << "\n";
    }

    node->setAttr(getActivityAnnotationAttrName(),
                  voMap.serialize(node.getContext()));
    os << "[ata] forward value origins:\n";
    for (ArrayAttr pair :
         node->getAttrOfType<ArrayAttr>(getActivityAnnotationAttrName())
             .getAsRange<ArrayAttr>()) {
      os << "     " << pair[0] << " originates from " << pair[1] << "\n";
    }

    auto *backwardOriginMap =
        solver.getOrCreateState<enzyme::BackwardOriginsMap>(
            &node.getCallableRegion()->front().front());
    Attribute backwardOrigins = backwardOriginMap->serialize(node.getContext());
    os << "[ata] backward value origins:\n";
    for (ArrayAttr pair :
         cast<ArrayAttr>(backwardOrigins).getAsRange<ArrayAttr>()) {
      os << "     " << pair[0] << " goes to " << pair[1] << "\n";
    }

    // Serialize return origins
    MLIRContext *ctx = node.getContext();
    SmallVector<Attribute> serializedReturnOperandOrigins(
        returnOperandOrigins.size());
    llvm::transform(returnOperandOrigins,
                    serializedReturnOperandOrigins.begin(),
                    [ctx](enzyme::ForwardOriginsLattice lattice) -> Attribute {
                      return lattice.serialize(ctx);
                    });
    node->setAttr(
        getReturnOriginsAttrName(),
        ArrayAttr::get(node.getContext(), serializedReturnOperandOrigins));
    os << "[ata] return origins: " << node->getAttr(getReturnOriginsAttrName())
       << "\n";

    node.getCallableRegion()->walk([&](Operation *op) {
      if (op->hasAttr("tag")) {
        for (OpResult result : op->getResults()) {
          auto *sources =
              solver.getOrCreateState<enzyme::ForwardOriginsLattice>(result);
          auto *sinks =
              solver.getOrCreateState<enzyme::BackwardOriginsLattice>(result);
          os << op->getAttr("tag") << "(#" << result.getResultNumber()
             << ") sources:\n"
             << *sources << "sinks:\n"
             << *sinks << "\n";
        }
      }
    });
  }
}
