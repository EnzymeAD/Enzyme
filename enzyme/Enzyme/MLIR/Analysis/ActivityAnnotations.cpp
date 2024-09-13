#include "ActivityAnnotations.h"
#include "DataFlowAliasAnalysis.h"
#include "DataFlowLattice.h"
#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"

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

static bool isPossiblyActive(Type type) {
  return isa<FloatType, ComplexType>(type);
}

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
  auto arg = dyn_cast<BlockArgument>(lattice->getAnchor());
  if (!arg) {
    assert(lattice->isUndefined());
    return;
  }
  if (!isPossiblyActive(arg.getType())) {
    return;
  }

  auto funcOp = cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
  auto origin = ArgumentOriginAttr::get(FlatSymbolRefAttr::get(funcOp),
                                        arg.getArgNumber());
  return propagateIfChanged(
      lattice, lattice->join(ForwardOriginsLattice::single(lattice->getAnchor(),
                                                           origin)));
}

/// True iff all results differentially depend on all operands
// TODO: differential dependency/activity interface
// TODO: Select cond is not fully active
static bool isFullyActive(Operation *op) {
  return isa<LLVM::FMulOp, LLVM::FAddOp, LLVM::FDivOp, LLVM::FSubOp,
             LLVM::FNegOp, LLVM::FAbsOp, LLVM::SqrtOp, LLVM::SinOp, LLVM::CosOp,
             LLVM::Exp2Op, LLVM::ExpOp, LLVM::LogOp, LLVM::InsertValueOp,
             LLVM::ExtractValueOp, LLVM::BitcastOp, LLVM::SelectOp>(op);
}

static bool isNoOp(Operation *op) {
  return isa<LLVM::NoAliasScopeDeclOp, LLVM::LifetimeStartOp,
             LLVM::LifetimeEndOp, LLVM::AssumeOp>(op);
}

LogicalResult enzyme::ForwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, ArrayRef<const ForwardOriginsLattice *> operands,
    ArrayRef<ForwardOriginsLattice *> results) {
  if (isFullyActive(op)) {
    for (ForwardOriginsLattice *result : results) {
      for (const ForwardOriginsLattice *operand : operands) {
        join(result, *operand);
      }
    }
    return success();
  }

  // Expected to be handled through the diff dependency interface
  if (isPure(op) || isNoOp(op))
    return success();

  auto markResultsUnknown = [&]() {
    for (ForwardOriginsLattice *result : results) {
      propagateIfChanged(result, result->markUnknown());
    }
  };

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    markResultsUnknown();
    return success();
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
  return success();
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
      if (isPossiblyActive(result->getAnchor().getType())) {
        propagateIfChanged(result,
                           result->merge(originsMap->getOrigins(srcClass)));
      }
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
    if (auto returnOriginsAttr = callee->getAttrOfType<ArrayAttr>(
            EnzymeDialect::getSparseActivityAnnotationAttrName())) {
      SmallVector<ValueOriginSet> returnOrigins;
      deserializeReturnOrigins(returnOriginsAttr, returnOrigins);
      return processCallToSummarizedFunc(call, returnOrigins, operands,
                                         results);
    }
  }

  // In the absence of a summary attribute, assume all results differentially
  // depend on all operands
  for (ForwardOriginsLattice *result : results)
    for (const ForwardOriginsLattice *operand : operands)
      join(result, *operand);
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
      auto *denseOrigins = getOrCreateFor<ForwardOriginsMap>(call, call);
      auto *pointsTo = getOrCreateFor<PointsToSets>(call, call);
      (void)returnOrigin.foreachElement(
          [&](OriginAttr calleeOrigin, ValueOriginSet::State state) {
            assert(state == ValueOriginSet::State::Defined &&
                   "undefined and unknown must have been handled above");
            auto calleeArgOrigin = cast<ArgumentOriginAttr>(calleeOrigin);
            // If the caller is a pointer, need to join what it points to
            const ForwardOriginsLattice *operandOrigins =
                operands[calleeArgOrigin.getArgNumber()];
            auto *callerAliasClass = getOrCreateFor<AliasClassLattice>(
                call, operandOrigins->getAnchor());
            traversePointsToSets(callerAliasClass->getAliasClassesObject(),
                                 *pointsTo, [&](DistinctAttr aliasClass) {
                                   (void)callerOrigins.join(
                                       denseOrigins->getOrigins(aliasClass));
                                 });
            return callerOrigins.join(operandOrigins->getOriginsObject());
          });
    }
    propagateIfChanged(result, result->merge(callerOrigins));
  }
}

void enzyme::BackwardActivityAnnotationAnalysis::setToExitState(
    BackwardOriginsLattice *lattice) {
  propagateIfChanged(lattice, lattice->markUnknown());
}

LogicalResult enzyme::BackwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, ArrayRef<BackwardOriginsLattice *> operands,
    ArrayRef<const BackwardOriginsLattice *> results) {
  if (isFullyActive(op)) {
    for (BackwardOriginsLattice *operand : operands)
      for (const BackwardOriginsLattice *result : results)
        meet(operand, *result);
    return success();
  }

  // Expected to be handled through the diff dependency interface
  if (isPure(op) || isNoOp(op))
    return success();

  auto markOperandsUnknown = [&]() {
    for (BackwardOriginsLattice *operand : operands) {
      propagateIfChanged(operand, operand->markUnknown());
    }
  };

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    markOperandsUnknown();
    return success();
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
  return success();
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
    if (auto returnOriginsAttr = callee->getAttrOfType<ArrayAttr>(
            EnzymeDialect::getSparseActivityAnnotationAttrName())) {
      SmallVector<ValueOriginSet> returnOrigins;
      deserializeReturnOrigins(returnOriginsAttr, returnOrigins);
      return processCallToSummarizedFunc(call, returnOrigins, operands,
                                         results);
    }
  }

  // In the absence of a summary attribute, assume all results differentially
  // depend on all operands
  for (BackwardOriginsLattice *operand : operands)
    for (const BackwardOriginsLattice *result : results)
      meet(operand, *result);
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
  auto *block = dyn_cast<Block *>(lattice->getAnchor().get<ProgramPoint>());
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

LogicalResult enzyme::DenseActivityAnnotationAnalysis::visitOperation(
    Operation *op, const ForwardOriginsMap &before, ForwardOriginsMap *after) {
  join(after, before);

  if (isNoOp(op))
    return success();

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    propagateIfChanged(after, after->markAllOriginsUnknown());
    return success();
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    if (!value) {
      propagateIfChanged(after, after->markAllOriginsUnknown());
      return success();
    }

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
        if (!isPossiblyActive(stored->getType())) {
          continue;
        }
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
  return success();
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
              EnzymeDialect::getDenseActivityAnnotationAttrName())) {
        DenseMap<DistinctAttr, ValueOriginSet> summary;
        deserializePointsTo(summaryAttr, summary);
        return processCallToSummarizedFunc(call, summary, before, after);
      }
    }
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
              EnzymeDialect::getDenseActivityAnnotationAttrName())) {
        DenseMap<DistinctAttr, ValueOriginSet> summary;
        deserializePointsTo(summaryAttr, summary);
        return processCallToSummarizedFunc(call, summary, after, before);
      }
    }
  }
}

void enzyme::DenseBackwardActivityAnnotationAnalysis::setToExitState(
    BackwardOriginsMap *lattice) {
  // FIXME: clean up how we access the (potential) block here
  auto *block = dyn_cast<Block *>(lattice->getAnchor().get<ProgramPoint>());
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

LogicalResult enzyme::DenseBackwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, const BackwardOriginsMap &after,
    BackwardOriginsMap *before) {
  meet(before, after);

  if (isNoOp(op))
    return success();

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    propagateIfChanged(before, before->markAllOriginsUnknown());
    return success();
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;

    Value value = effect.getValue();
    if (!value) {
      propagateIfChanged(before, before->markAllOriginsUnknown());
      return success();
    }

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
  return success();
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

// TODO: the alias summary attribute is sufficent to get the correct behaviour
// here, but it would be nice if these were not hardcoded.
void annotateHardcoded(FunctionOpInterface func) {
  if (func.getName() == "lgamma" || func.getName() == "tanh") {
    MLIRContext *ctx = func.getContext();
    SmallVector<Attribute> arr = {StringAttr::get(ctx, "<undefined>")};
    func->setAttr(enzyme::EnzymeDialect::getAliasSummaryAttrName(),
                  ArrayAttr::get(ctx, arr));
  }
}

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

using OriginsPair =
    std::pair<enzyme::ForwardOriginsLattice, enzyme::BackwardOriginsLattice>;

/// Once having reached a top-level entry point, go top-down and convert the
/// relative sources/sinks into concrete active/constant results.
///
/// This would ideally be done after lowering to LLVM and during differentiation
/// because it loses context sensitivity, but this is faster to prototype with.
void topDownActivityAnalysis(
    FunctionOpInterface callee, ArrayRef<enzyme::Activity> argActivities,
    ArrayRef<enzyme::Activity> retActivities,
    DenseMap<BlockArgument, OriginsPair> &blockArgOrigins) {
  using namespace mlir::enzyme;
  MLIRContext *ctx = callee.getContext();
  callee->setAttr("enzyme.visited", UnitAttr::get(ctx));
  auto trueAttr = BoolAttr::get(ctx, true);
  auto falseAttr = BoolAttr::get(ctx, false);

  auto isOriginActive = [&](OriginAttr origin) {
    if (auto argOriginAttr = dyn_cast<ArgumentOriginAttr>(origin)) {
      return llvm::is_contained({Activity::enzyme_dup,
                                 Activity::enzyme_dupnoneed,
                                 Activity::enzyme_active},
                                argActivities[argOriginAttr.getArgNumber()]);
    }
    auto retOriginAttr = cast<ReturnOriginAttr>(origin);
    return llvm::is_contained({Activity::enzyme_dup, Activity::enzyme_dupnoneed,
                               Activity::enzyme_active},
                              retActivities[retOriginAttr.getReturnNumber()]);
  };
  callee.getFunctionBody().walk([&](Operation *op) {
    if (op->getNumResults() == 0) {
      // Operations that don't return values are definitionally "constant"
      op->setAttr("enzyme.icv", trueAttr);
    } else {
      // Value activity
      if (op->hasAttr("enzyme.constantval")) {
        op->setAttr("enzyme.icv", trueAttr);
      } else if (op->hasAttr("enzyme.activeval")) {
        op->setAttr("enzyme.icv", falseAttr);
      } else {
        auto valueSource = op->getAttrOfType<ArrayAttr>("enzyme.valsrc");
        auto valueSink = op->getAttrOfType<ArrayAttr>("enzyme.valsink");
        if (!(valueSource && valueSink)) {
          llvm::errs() << "[activity] missing attributes for op: " << *op
                       << "\n";
        }
        assert(valueSource && valueSink && "missing attributes for op");
        bool activeSource =
            llvm::any_of(valueSource.getAsRange<OriginAttr>(), isOriginActive);
        bool activeSink =
            llvm::any_of(valueSink.getAsRange<OriginAttr>(), isOriginActive);
        bool activeVal = activeSource && activeSink;
        op->setAttr("enzyme.icv", BoolAttr::get(ctx, !activeVal));
      }
    }
    op->removeAttr("enzyme.constantval");
    op->removeAttr("enzyme.activeval");
    op->removeAttr("enzyme.valsrc");
    op->removeAttr("enzyme.valsink");

    // Instruction activity
    if (op->hasAttr("enzyme.constantop")) {
      op->setAttr("enzyme.ici", trueAttr);
    } else if (op->hasAttr("enzyme.activeop")) {
      op->setAttr("enzyme.ici", falseAttr);
    } else {
      bool activeSource = llvm::any_of(
          op->getAttrOfType<ArrayAttr>("enzyme.opsrc").getAsRange<OriginAttr>(),
          isOriginActive);
      bool activeSink =
          llvm::any_of(op->getAttrOfType<ArrayAttr>("enzyme.opsink")
                           .getAsRange<OriginAttr>(),
                       isOriginActive);
      bool activeOp = activeSource && activeSink;
      op->setAttr("enzyme.ici", BoolAttr::get(ctx, !activeOp));
    }

    op->removeAttr("enzyme.constantop");
    op->removeAttr("enzyme.activeop");
    op->removeAttr("enzyme.opsrc");
    op->removeAttr("enzyme.opsink");

    if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      auto funcOp = cast<FunctionOpInterface>(callOp.resolveCallable());
      if (!funcOp->hasAttr("enzyme.visited")) {
        SmallVector<Activity> callArgActivities, callResActivities;
        for (Value operand : callOp.getArgOperands()) {
          if (auto *definingOp = operand.getDefiningOp()) {
            bool icv =
                definingOp->getAttrOfType<BoolAttr>("enzyme.icv").getValue();
            callArgActivities.push_back(icv ? Activity::enzyme_const
                                            : Activity::enzyme_active);
          } else {
            BlockArgument blockArg = cast<BlockArgument>(operand);
            const OriginsPair &originsPair = blockArgOrigins.at(blockArg);
            const ForwardOriginsLattice &sources = originsPair.first;
            const BackwardOriginsLattice &sinks = originsPair.second;
            bool argActive = false;
            if (sources.isUnknown() || sinks.isUnknown()) {
              argActive = true;
            } else if (sources.isUndefined() || sinks.isUndefined()) {
              argActive = false;
            } else {
              argActive = llvm::any_of(sources.getOrigins(), isOriginActive) &&
                          llvm::any_of(sinks.getOrigins(), isOriginActive);
            }
            callArgActivities.push_back(argActive ? Activity::enzyme_active
                                                  : Activity::enzyme_const);
          }
        }
        if (op->getNumResults() != 0) {
          bool icv = op->getAttrOfType<BoolAttr>("enzyme.icv").getValue();
          callResActivities.push_back(icv ? Activity::enzyme_const
                                          : Activity::enzyme_active);
        }

        topDownActivityAnalysis(funcOp, callArgActivities, callResActivities,
                                blockArgOrigins);
      }
    }
  });
}
} // namespace

void enzyme::runActivityAnnotations(
    FunctionOpInterface callee, ArrayRef<enzyme::Activity> argActivities,
    const ActivityPrinterConfig &activityConfig) {
  SymbolTableCollection symbolTable;
  SmallVector<CallableOpInterface> sorted;
  reverseToposortCallgraph(callee, &symbolTable, sorted);
  raw_ostream &os = llvm::outs();

  // TODO: is there any way of serializing information in a block argument?
  DenseMap<BlockArgument, OriginsPair> blockArgOrigins;

  StringRef pointerSummaryName = EnzymeDialect::getPointerSummaryAttrName();
  for (CallableOpInterface node : sorted) {
    annotateHardcoded(cast<FunctionOpInterface>(node.getOperation()));

    if (!node.getCallableRegion() || node->hasAttr(pointerSummaryName))
      continue;
    auto funcOp = cast<FunctionOpInterface>(node.getOperation());
    if (activityConfig.verbose) {
      os << "[ata] processing function @" << funcOp.getName() << "\n";
    }
    DataFlowConfig dataFlowConfig;
    dataFlowConfig.setInterprocedural(false);
    DataFlowSolver solver(dataFlowConfig);
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
    enzyme::ForwardOriginsMap forwardOriginsMap(nullptr);
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
          (void)forwardOriginsMap.join(*returnOrigins);

        for (OpOperand &operand : op.getOpOperands()) {
          (void)returnAliasClasses[operand.getOperandNumber()].join(
              *solver.lookupState<enzyme::AliasClassLattice>(operand.get()));
          (void)returnOperandOrigins[operand.getOperandNumber()].join(
              *solver.lookupState<enzyme::ForwardOriginsLattice>(
                  operand.get()));
        }
      }
    }

    // Sparse alias annotations
    SmallVector<Attribute> aliasAttributes(returnAliasClasses.size());
    llvm::transform(returnAliasClasses, aliasAttributes.begin(),
                    [&](enzyme::AliasClassLattice lattice) {
                      return lattice.serialize(node.getContext());
                    });
    node->setAttr(EnzymeDialect::getAliasSummaryAttrName(),
                  ArrayAttr::get(node.getContext(), aliasAttributes));

    // Points-to-pointer annotations
    node->setAttr(pointerSummaryName, p2sets.serialize(node.getContext()));
    if (activityConfig.verbose) {
      os << "[ata] p2p summary:\n";
      if (node->getAttrOfType<ArrayAttr>(pointerSummaryName).size() == 0) {
        os << "     <empty>\n";
      }
      for (ArrayAttr pair : node->getAttrOfType<ArrayAttr>(pointerSummaryName)
                                .getAsRange<ArrayAttr>()) {
        os << "     " << pair[0] << " -> " << pair[1] << "\n";
      }
    }

    node->setAttr(EnzymeDialect::getDenseActivityAnnotationAttrName(),
                  forwardOriginsMap.serialize(node.getContext()));
    if (activityConfig.verbose) {
      os << "[ata] forward value origins:\n";
      for (ArrayAttr pair :
           node->getAttrOfType<ArrayAttr>(
                   EnzymeDialect::getDenseActivityAnnotationAttrName())
               .getAsRange<ArrayAttr>()) {
        os << "     " << pair[0] << " originates from " << pair[1] << "\n";
      }
    }

    auto *backwardOriginsMap =
        solver.getOrCreateState<enzyme::BackwardOriginsMap>(
            &node.getCallableRegion()->front().front());
    Attribute backwardOrigins =
        backwardOriginsMap->serialize(node.getContext());
    if (activityConfig.verbose) {
      os << "[ata] backward value origins:\n";
      for (ArrayAttr pair :
           cast<ArrayAttr>(backwardOrigins).getAsRange<ArrayAttr>()) {
        os << "     " << pair[0] << " goes to " << pair[1] << "\n";
      }
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
        EnzymeDialect::getSparseActivityAnnotationAttrName(),
        ArrayAttr::get(node.getContext(), serializedReturnOperandOrigins));
    if (activityConfig.verbose) {
      os << "[ata] return origins: "
         << node->getAttr(EnzymeDialect::getSparseActivityAnnotationAttrName())
         << "\n";
    }

    auto joinActiveDataState =
        [&](Value value,
            std::pair<ForwardOriginsLattice, BackwardOriginsLattice> &out) {
          auto *sources = solver.getOrCreateState<ForwardOriginsLattice>(value);
          auto *sinks = solver.getOrCreateState<BackwardOriginsLattice>(value);
          (void)out.first.join(*sources);
          (void)out.second.meet(*sinks);
        };

    auto joinActivePointerState =
        [&](const AliasClassSet &aliasClasses,
            std::pair<ForwardOriginsLattice, BackwardOriginsLattice> &out) {
          traversePointsToSets(
              aliasClasses, p2sets, [&](DistinctAttr aliasClass) {
                (void)out.first.merge(forwardOriginsMap.getOrigins(aliasClass));
                (void)out.second.merge(
                    backwardOriginsMap->getOrigins(aliasClass));
              });
        };

    auto joinActiveValueState =
        [&](Value value,
            std::pair<ForwardOriginsLattice, BackwardOriginsLattice> &out) {
          if (isa<LLVM::LLVMPointerType, MemRefType>(value.getType())) {
            auto *aliasClasses =
                solver.getOrCreateState<AliasClassLattice>(value);
            joinActivePointerState(aliasClasses->getAliasClassesObject(), out);
          } else {
            joinActiveDataState(value, out);
          }
        };

    auto annotateActivity = [&](Operation *op) {
      assert(op->getNumResults() < 2 && op->getNumRegions() == 0 &&
             "annotation only supports the LLVM dialect");
      auto unitAttr = UnitAttr::get(ctx);
      // Check activity of values
      for (OpResult result : op->getResults()) {
        std::pair<ForwardOriginsLattice, BackwardOriginsLattice>
            activityAttributes({result, ValueOriginSet()},
                               {result, ValueOriginSet()});
        joinActiveValueState(result, activityAttributes);
        const auto &sources = activityAttributes.first;
        const auto &sinks = activityAttributes.second;
        // Possible states: if either source or sink is undefined or empty, the
        // value is always constant.
        if (sources.isUnknown() || sinks.isUnknown()) {
          // Always active
          op->setAttr("enzyme.activeval", unitAttr);
        } else if (sources.isUndefined() || sinks.isUndefined()) {
          // Always constant
          op->setAttr("enzyme.constantval", unitAttr);
        } else {
          // Conditionally active depending on the activity of sources and sinks
          op->setAttr("enzyme.valsrc", sources.serialize(ctx));
          op->setAttr("enzyme.valsink", sinks.serialize(ctx));
        }
      }
      // Check activity of operation
      StringRef opSourceAttrName = "enzyme.opsrc";
      StringRef opSinkAttrName = "enzyme.opsink";
      std::pair<ForwardOriginsLattice, BackwardOriginsLattice> opAttributes(
          {nullptr, ValueOriginSet()}, {nullptr, ValueOriginSet()});
      if (isPure(op)) {
        // A pure operation can only propagate data via its results
        for (OpResult result : op->getResults()) {
          joinActiveDataState(result, opAttributes);
        }
      } else {
        // We need a special case because stores of active pointers don't fit
        // the definition but are active instructions
        if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
          auto *storedClass =
              solver.getOrCreateState<AliasClassLattice>(storeOp.getValue());
          joinActivePointerState(storedClass->getAliasClassesObject(),
                                 opAttributes);
        } else if (auto callOp = dyn_cast<CallOpInterface>(op)) {
          // TODO: tricky, requires some thought
          auto callable = cast<CallableOpInterface>(callOp.resolveCallable());
          if (callable->hasAttr(
                  EnzymeDialect::getDenseActivityAnnotationAttrName())) {
            for (Value operand : callOp.getArgOperands())
              joinActiveValueState(operand, opAttributes);
          }
          // We need to
          // determine if the body of the function contains active instructions
        }

        // Default: the op is active iff any of its operands or results are
        // active data.
        for (Value operand : op->getOperands())
          joinActiveDataState(operand, opAttributes);
        for (OpResult result : op->getResults())
          joinActiveDataState(result, opAttributes);
      }

      const auto &opSources = opAttributes.first;
      const auto &opSinks = opAttributes.second;
      if (opSources.isUnknown() || opSinks.isUnknown()) {
        op->setAttr("enzyme.activeop", unitAttr);
      } else if (opSources.isUndefined() || opSinks.isUndefined()) {
        op->setAttr("enzyme.constantop", unitAttr);
      } else {
        op->setAttr(opSourceAttrName, opAttributes.first.serialize(ctx));
        op->setAttr(opSinkAttrName, opAttributes.second.serialize(ctx));
      }
    };

    // We lose the solver state when going top down and I don't know a better
    // way to serialize block argument information.
    node.getCallableRegion()->walk([&](Block *block) {
      for (BlockArgument blockArg : block->getArguments()) {
        OriginsPair blockArgAttributes({blockArg, ValueOriginSet()},
                                       {blockArg, ValueOriginSet()});
        joinActiveValueState(blockArg, blockArgAttributes);
        blockArgOrigins.try_emplace(blockArg, blockArgAttributes);
      }
    });

    node.getCallableRegion()->walk([&](Operation *op) {
      if (activityConfig.annotate)
        annotateActivity(op);
      if (activityConfig.verbose) {
        if (op->hasAttr("tag")) {
          for (OpResult result : op->getResults()) {
            std::pair<ForwardOriginsLattice, BackwardOriginsLattice>
                activityAttributes({result, ValueOriginSet()},
                                   {result, ValueOriginSet()});
            joinActiveValueState(result, activityAttributes);
            os << op->getAttr("tag") << "(#" << result.getResultNumber()
               << ")\n"
               << "  sources: " << activityAttributes.first.serialize(ctx)
               << "\n"
               << "  sinks:   " << activityAttributes.second.serialize(ctx)
               << "\n";
          }
        }
      }
    });
  }

  if (!argActivities.empty() && activityConfig.annotate) {
    SmallVector<enzyme::Activity> resActivities;
    for (Type resultType : callee.getResultTypes()) {
      resActivities.push_back(isa<FloatType, ComplexType>(resultType)
                                  ? Activity::enzyme_active
                                  : Activity::enzyme_const);
    }

    topDownActivityAnalysis(callee, argActivities, resActivities,
                            blockArgOrigins);
  }
}
