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

raw_ostream &enzyme::operator<<(raw_ostream &os,
                                const enzyme::AliasClassSet &aliasClassSet) {
  aliasClassSet.print(os);
  return os;
}

void enzyme::ValueOriginsLattice::print(raw_ostream &os) const {
  if (isUnknown()) {
    os << "Unknown VO";
  } else if (isUndefined()) {
    os << "Undefined VO";
  } else {
    os << "size: " << origins.getElements().size() << ":\n";
    for (auto aliasClass : origins.getElements()) {
      os << "  " << aliasClass << "\n";
    }
  }
}

ChangeResult
enzyme::ValueOriginsLattice::join(const AbstractSparseLattice &other) {
  const auto *otherValueOrigins =
      static_cast<const ValueOriginsLattice *>(&other);
  return origins.join(otherValueOrigins->origins);
}

void enzyme::ForwardActivityAnnotationAnalysis::setToEntryState(
    ValueOriginsLattice *lattice) {
  auto arg = dyn_cast<BlockArgument>(lattice->getPoint());
  if (!arg) {
    assert(lattice->isUndefined());
    return;
  }

  auto funcOp = cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
  auto origin = ArgumentOriginAttr::get(FlatSymbolRefAttr::get(funcOp),
                                        arg.getArgNumber());
  return propagateIfChanged(lattice, lattice->join(ValueOriginsLattice::single(
                                         lattice->getPoint(), origin)));
}

void enzyme::ForwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, ArrayRef<const ValueOriginsLattice *> operands,
    ArrayRef<ValueOriginsLattice *> results) {
  // TODO: Differential dependency/activity interface
  if (isa<LLVM::FMulOp, LLVM::FAddOp, LLVM::FDivOp, LLVM::FSubOp, LLVM::FNegOp,
          LLVM::FAbsOp, LLVM::SqrtOp, LLVM::SinOp, LLVM::CosOp, LLVM::Exp2Op,
          LLVM::ExpOp, LLVM::InsertValueOp, LLVM::ExtractValueOp,
          LLVM::BitcastOp>(op)) {
    // All results differentially depend on all operands
    for (ValueOriginsLattice *result : results) {
      for (const ValueOriginsLattice *operand : operands) {
        join(result, *operand);
      }
    }
    return;
  }

  // Expected to be handled through the diff dependency interface
  if (isPure(op))
    return;

  if (isa<LLVM::NoAliasScopeDeclOp, LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(
          op))
    return;

  auto markResultsUnknown = [&]() {
    for (ValueOriginsLattice *result : results) {
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
    Operation *op, Value address, ArrayRef<ValueOriginsLattice *> results) {
  auto markResultsUnknown = [&]() {
    for (ValueOriginsLattice *result : results) {
      propagateIfChanged(result, result->markUnknown());
    }
  };

  auto *srcClasses = getOrCreateFor<AliasClassLattice>(op, address);
  auto *originsMap = getOrCreateFor<ValueOriginsMap>(op, op);
  if (srcClasses->isUndefined())
    return;
  if (srcClasses->isUnknown())
    return markResultsUnknown();

  // Look up the alias class and see what its origins are, then propagate
  // those origins to the read results.
  for (DistinctAttr srcClass : srcClasses->getAliasClasses()) {
    const ValueOriginSet &origins = originsMap->getOrigins(srcClass);
    if (origins.isUndefined())
      continue;
    if (origins.isUnknown())
      return markResultsUnknown();

    for (ValueOriginsLattice *result : results) {
      propagateIfChanged(result, result->insert(origins.getElements()));
    }
  }
}

void enzyme::ForwardActivityAnnotationAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const ValueOriginsLattice *> operands,
    ArrayRef<ValueOriginsLattice *> results) {}

void enzyme::ValueOriginsMap::print(raw_ostream &os) const {
  if (valueOrigins.empty()) {
    os << "<empty>\n";
    return;
  }
  for (const auto &[aliasClass, origins] : valueOrigins) {
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

static bool sortKeys(Attribute a, Attribute b) {
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

// TODO(jacob): reduce code duplication (again)
Attribute enzyme::ValueOriginsMap::serialize(MLIRContext *ctx) const {
  SmallVector<Attribute> pointsToArray;

  for (const auto &[srcClass, destClasses] : valueOrigins) {
    SmallVector<Attribute, 2> pair = {srcClass};
    SmallVector<Attribute, 5> aliasClasses;
    if (destClasses.isUnknown()) {
      aliasClasses.push_back(StringAttr::get(ctx, "unknown"));
    } else if (destClasses.isUndefined()) {
      aliasClasses.push_back(StringAttr::get(ctx, "undefined"));
    } else {
      for (const ArgumentOriginAttr &destClass : destClasses.getElements()) {
        aliasClasses.push_back(destClass);
      }
      llvm::sort(aliasClasses, sortKeys);
    }
    pair.push_back(ArrayAttr::get(ctx, aliasClasses));
    pointsToArray.push_back(ArrayAttr::get(ctx, pair));
  }
  llvm::sort(pointsToArray, [&](Attribute a, Attribute b) {
    auto arrA = cast<ArrayAttr>(a);
    auto arrB = cast<ArrayAttr>(b);
    return sortKeys(arrA[0], arrB[0]);
  });
  return ArrayAttr::get(ctx, pointsToArray);
}

ChangeResult enzyme::ValueOriginsMap::join(const AbstractDenseLattice &other) {
  const auto &rhs = static_cast<const ValueOriginsMap &>(other);
  llvm::SmallDenseSet<DistinctAttr> keys;
  auto lhsRange = llvm::make_first_range(valueOrigins);
  auto rhsRange = llvm::make_first_range(rhs.valueOrigins);
  keys.insert(lhsRange.begin(), lhsRange.end());
  keys.insert(rhsRange.begin(), rhsRange.end());

  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr key : keys) {
    auto lhsIt = valueOrigins.find(key);
    auto rhsIt = rhs.valueOrigins.find(key);
    assert(lhsIt != valueOrigins.end() || rhsIt != rhs.valueOrigins.end());

    // If present in both, join.
    if (lhsIt != valueOrigins.end() && rhsIt != rhs.valueOrigins.end()) {
      result |= lhsIt->getSecond().join(rhsIt->getSecond());
      continue;
    }

    // Copy from RHS if available only there.
    if (lhsIt == valueOrigins.end()) {
      valueOrigins.try_emplace(rhsIt->getFirst(), rhsIt->getSecond());
      result = ChangeResult::Change;
    }

    // Do nothing if available only in LHS.
  }
  return result;
}

ChangeResult enzyme::ValueOriginsMap::insert(const AliasClassSet &keysToUpdate,
                                             const ValueOriginSet &origins) {
  if (keysToUpdate.isUnknown())
    return markAllOriginsUnknown();

  if (keysToUpdate.isUndefined())
    return ChangeResult::NoChange;

  return keysToUpdate.foreachClass(
      [&](DistinctAttr key, AliasClassSet::State state) {
        assert(state == AliasClassSet::State::Defined &&
               "unknown must have been handled above");
        return joinPotentiallyMissing(key, origins);
      });
}

ChangeResult enzyme::ValueOriginsMap::markAllOriginsUnknown() {
  ChangeResult result = ChangeResult::NoChange;
  for (auto &it : valueOrigins)
    result |= it.getSecond().join(ValueOriginSet::getUnknown());
  return result;
}

ChangeResult
enzyme::ValueOriginsMap::joinPotentiallyMissing(DistinctAttr key,
                                                const ValueOriginSet &value) {
  // Don't store explicitly undefined values in the mapping, keys absent from
  // the mapping are treated as implicitly undefined.
  if (value.isUndefined())
    return ChangeResult::NoChange;

  bool inserted;
  decltype(valueOrigins.begin()) iterator;
  std::tie(iterator, inserted) = valueOrigins.try_emplace(key, value);
  if (!inserted)
    return iterator->second.join(value);
  return ChangeResult::Change;
}

void enzyme::DenseActivityAnnotationAnalysis::setToEntryState(
    ValueOriginsMap *lattice) {
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
    Operation *op, const ValueOriginsMap &before, ValueOriginsMap *after) {
  join(after, before);

  if (isa<LLVM::NoAliasScopeDeclOp, LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(
          op))
    return;

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    return propagateIfChanged(after, after->markAllOriginsUnknown());
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    // TODO: may be too pessimistic
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
        auto *origins = getOrCreateFor<ValueOriginsLattice>(op, *stored);
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
    const ValueOriginsMap &before, ValueOriginsMap *after) {
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
      auto pointsTo =
          cast<ArrayAttr>(pair[1]).getAsRange<enzyme::ArgumentOriginAttr>();
      // TODO: see if there's a nice way to convert the
      // AliasClassSet::insert method to accept this iterator rather than
      // constructing a DenseSet
      (void)pointsToSet.insert(DenseSet<enzyme::ArgumentOriginAttr>(
          pointsTo.begin(), pointsTo.end()));
    }

    summaryMap.insert({pointer, pointsToSet});
  }
}

void enzyme::DenseActivityAnnotationAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const ValueOriginsMap &before, ValueOriginsMap *after) {
  join(after, before);
  if (action == dataflow::CallControlFlowAction::ExternalCallee) {
    auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
    if (!symbol)
      return propagateIfChanged(after, after->markAllOriginsUnknown());

    if (auto callee = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
            call, symbol.getLeafReference())) {
      if (auto summaryAttr = callee->getAttrOfType<ArrayAttr>("activedeps")) {
        DenseMap<DistinctAttr, ValueOriginSet> summary;
        deserializePointsTo(summaryAttr, summary);
        return processCallToSummarizedFunc(call, summary, before, after);
      }
    }
  }
}

void enzyme::DenseActivityAnnotationAnalysis::processCallToSummarizedFunc(
    CallOpInterface call, const DenseMap<DistinctAttr, ValueOriginSet> &summary,
    const ValueOriginsMap &before, ValueOriginsMap *after) {
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
          getOrCreateFor<ValueOriginsLattice>(call, argOperand);
      (void)argOrigins.join(sparseOrigins->getOriginsObject());
    } else {
      // Unify all the origins
      // Since we're not keeping track of argument depth, we need to union the
      // arg origins with everything it points to.
      AliasClassSet current = argClasses->getAliasClassesObject();
      while (!current.isUndefined()) {
        AliasClassSet next;
        for (DistinctAttr currentClass : current.getAliasClasses()) {
          (void)argOrigins.join(before.getOrigins(currentClass));
          (void)next.join(p2sets->getPointsTo(currentClass));
        }
        std::swap(current, next);
      }
    }
    argumentClasses.push_back(argClasses->getAliasClassesObject());
    argumentOrigins.push_back(argOrigins);
  }

  // TODO: Does the traversal order matter here?
  for (const auto &[destClass, sourceOrigins] : summary) {
    ValueOriginSet callerOrigins;
    for (ArgumentOriginAttr sourceOrigin : sourceOrigins.getElements()) {
      unsigned argNumber = sourceOrigin.getArgNumber();
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
        for (DistinctAttr currentClass : current.getAliasClasses())
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
} // namespace

void enzyme::runActivityAnnotations(FunctionOpInterface callee) {
  SymbolTableCollection symbolTable;
  SmallVector<CallableOpInterface> sorted;
  reverseToposortCallgraph(callee, &symbolTable, sorted);
  raw_ostream &os = llvm::outs();

  for (CallableOpInterface node : sorted) {
    if (!node.getCallableRegion() || node->hasAttr("p2psummary"))
      continue;
    auto funcOp = cast<FunctionOpInterface>(node.getOperation());
    os << "[ata] processing function @" << funcOp.getName() << "\n";
    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);

    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<enzyme::AliasAnalysis>(callee.getContext(),
                                       /*relative=*/true);
    solver.load<enzyme::PointsToPointerAnalysis>();
    solver.load<enzyme::ForwardActivityAnnotationAnalysis>();
    solver.load<enzyme::DenseActivityAnnotationAnalysis>();

    if (failed(solver.initializeAndRun(node))) {
      assert(false && "dataflow solver failed");
    }

    // Create the overall summary by joining sets at all return sites.
    enzyme::PointsToSets p2sets(nullptr);
    enzyme::ValueOriginsMap voMap(nullptr);
    size_t numResults = node.getResultTypes().size();
    SmallVector<enzyme::ValueOriginsLattice> returnOperandOrigins(
        numResults, ValueOriginsLattice(nullptr));
    SmallVector<enzyme::AliasClassLattice> returnAliasClasses(
        numResults, AliasClassLattice(nullptr));

    raw_ostream &os = llvm::outs();
    for (Operation &op : node.getCallableRegion()->getOps()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) {
        (void)p2sets.join(*solver.lookupState<enzyme::PointsToSets>(&op));
        auto *returnOrigins = solver.lookupState<enzyme::ValueOriginsMap>(&op);
        if (returnOrigins)
          (void)voMap.join(*returnOrigins);

        for (OpOperand &operand : op.getOpOperands()) {
          (void)returnAliasClasses[operand.getOperandNumber()].join(
              *solver.lookupState<enzyme::AliasClassLattice>(operand.get()));
          (void)returnOperandOrigins[operand.getOperandNumber()].join(
              *solver.lookupState<enzyme::ValueOriginsLattice>(operand.get()));
        }
      }
    }

    for (auto lattice : returnAliasClasses) {
      os << "[debug] return alias class: " << lattice << "\n";
    }

    node->setAttr("p2psummary", p2sets.serialize(node.getContext()));
    os << "[ata] p2p summary:\n";
    if (node->getAttrOfType<ArrayAttr>("p2psummary").size() == 0) {
      os << "     <empty>\n";
    }
    for (ArrayAttr pair :
         node->getAttrOfType<ArrayAttr>("p2psummary").getAsRange<ArrayAttr>()) {
      os << "     " << pair[0] << " -> " << pair[1] << "\n";
    }

    node->setAttr("activedeps", voMap.serialize(node.getContext()));
    os << "[ata] value origin summary:\n";
    for (ArrayAttr pair :
         node->getAttrOfType<ArrayAttr>("activedeps").getAsRange<ArrayAttr>()) {
      os << "     " << pair[0] << " originates from " << pair[1] << "\n";
    }

    // Serialize return origins
    MLIRContext *ctx = node.getContext();
    SmallVector<Attribute> serializedReturnOperandOrigins(
        returnOperandOrigins.size());
    llvm::transform(
        returnOperandOrigins, serializedReturnOperandOrigins.begin(),
        [ctx](enzyme::ValueOriginsLattice lattice) -> Attribute {
          if (lattice.isUndefined())
            return StringAttr::get(ctx, "<undefined>");
          if (lattice.isUnknown())
            return StringAttr::get(ctx, "<unknown>");
          SmallVector<Attribute> originsVector(lattice.getOrigins().begin(),
                                               lattice.getOrigins().end());
          llvm::sort(originsVector, sortKeys);
          return ArrayAttr::get(ctx, originsVector);
        });
    node->setAttr(
        "returnorigins",
        ArrayAttr::get(node.getContext(), serializedReturnOperandOrigins));
    os << "[ata] return origins: " << node->getAttr("returnorigins") << "\n";
  }
}
