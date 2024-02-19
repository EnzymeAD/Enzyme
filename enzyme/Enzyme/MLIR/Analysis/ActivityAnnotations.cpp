#include "ActivityAnnotations.h"
#include "AliasAnalysis.h"
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

void enzyme::ValueOriginsLattice::print(raw_ostream &os) const {
  if (isUnknown()) {
    os << "Unknown VO";
  } else if (isUndefined()) {
    os << "Undefined VO";
  } else {
    os << "size: " << origins.getAliasClasses().size() << ":\n";
    for (auto aliasClass : origins.getAliasClasses()) {
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
  DistinctAttr argClass =
      originalClasses.getOriginalClass(lattice->getPoint(), origin);
  return propagateIfChanged(lattice, lattice->join(ValueOriginsLattice::single(
                                         lattice->getPoint(), argClass)));
}

void enzyme::ForwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, ArrayRef<const ValueOriginsLattice *> operands,
    ArrayRef<ValueOriginsLattice *> results) {
  // TODO: Differential dependency/activity interface
  if (isa<LLVM::FMulOp, LLVM::FAddOp, LLVM::FDivOp, LLVM::FSubOp, LLVM::FNegOp,
          LLVM::FAbsOp, LLVM::SqrtOp, LLVM::SinOp, LLVM::CosOp, LLVM::Exp2Op,
          LLVM::ExpOp>(op)) {
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

  if (isa<LLVM::NoAliasScopeDeclOp>(op))
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
    if (!value)
      markResultsUnknown();

    // we need to know if what this pointer points to has a value origin
    // dependency
    auto *srcClasses = getOrCreateFor<AliasClassLattice>(op, value);
    auto *originsMap = getOrCreateFor<ValueOriginsMap>(op, op);
    if (srcClasses->isUndefined())
      continue;
    if (srcClasses->isUnknown()) {
      markResultsUnknown();
      continue;
    }

    // Look up the alias class and see what its origins are, then propagate
    // those origins to the read results.
    for (DistinctAttr srcClass : srcClasses->getAliasClasses()) {
      const AliasClassSet &origins = originsMap->getOrigins(srcClass);
      for (ValueOriginsLattice *result : results) {
        if (origins.isUnknown())
          propagateIfChanged(result, result->markUnknown());
        if (origins.isUndefined())
          continue;
        propagateIfChanged(result, result->insert(origins.getAliasClasses()));
      }
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
      llvm::interleaveComma(origins.getAliasClasses(), os);
    }
    os << "}\n";
  }
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
                                             const AliasClassSet &origins) {
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
    result |= it.getSecond().join(AliasClassSet::getUnknown());
  return result;
}

ChangeResult
enzyme::ValueOriginsMap::joinPotentiallyMissing(DistinctAttr key,
                                                const AliasClassSet &value) {
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
    DistinctAttr argOrigin = originalClasses.getOriginalClass(arg, origin);
    changed |= lattice->insert(argClass->getAliasClassesObject(),
                               AliasClassSet(argOrigin));
  }
  propagateIfChanged(lattice, changed);
}

std::optional<Value> getStored(Operation *op);
std::optional<Value> getCopySource(Operation *op);

void enzyme::DenseActivityAnnotationAnalysis::visitOperation(
    Operation *op, const ValueOriginsMap &before, ValueOriginsMap *after) {
  join(after, before);

  if (isa<LLVM::NoAliasScopeDeclOp>(op))
    return;

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory) {
    return propagateIfChanged(after, after->markAllOriginsUnknown());
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;

    Value value = effect.getValue();
    // TODO: may be too pessimistic
    if (!value)
      return propagateIfChanged(after, after->markAllOriginsUnknown());

    if (std::optional<Value> stored = getStored(op)) {
      auto *origins = getOrCreateFor<ValueOriginsLattice>(op, *stored);
      auto *dest = getOrCreateFor<AliasClassLattice>(op, value);
      propagateIfChanged(after, after->insert(dest->getAliasClassesObject(),
                                              origins->getOriginsObject()));
    } else if (std::optional<Value> copySource = getCopySource(op)) {
    } else {
      return propagateIfChanged(after, after->markAllOriginsUnknown());
    }
  }
}

void enzyme::DenseActivityAnnotationAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const ValueOriginsMap &before, ValueOriginsMap *after) {
  join(after, before);
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
    raw_ostream &os = llvm::outs();
    for (Operation &op : node.getCallableRegion()->getOps()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) {
        (void)p2sets.join(*solver.lookupState<enzyme::PointsToSets>(&op));
        (void)voMap.join(*solver.lookupState<enzyme::ValueOriginsMap>(&op));

        for (OpOperand &operand : op.getOpOperands()) {
          os << "[aaa] return at idx " << operand.getOperandNumber()
             << " has value origin "
             << *solver.lookupState<enzyme::ValueOriginsLattice>(operand.get())
             << "\n";
        }
      }
    }

    node->setAttr("p2psummary", p2sets.serialize(node.getContext()));
    os << "[ata] p2p summary:\n";
    for (ArrayAttr pair :
         node->getAttrOfType<ArrayAttr>("p2psummary").getAsRange<ArrayAttr>()) {
      os << "     " << pair[0] << " -> " << pair[1] << "\n";
    }

    os << "[ata] vo summary:\n" << voMap << "\n";
  }
}
