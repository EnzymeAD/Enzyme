#include "Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

using namespace mlir;
using namespace mlir::dataflow;

struct ValueOrigin {
  ValueOrigin(int32_t position, int32_t depth)
      : position(position), depth(depth) {}

  bool operator<(const ValueOrigin &other) const {
    return position < other.position ||
           (position == other.position && depth < other.depth);
  }

  bool operator==(const ValueOrigin &other) const {
    return position == other.position && depth == other.depth;
  }

  void print(llvm::raw_ostream &os) const {
    os << "(" << position << ", " << depth << ")";
  }
  LLVM_DUMP_METHOD void dump() { print(llvm::errs()); }

  int32_t position;
  int32_t depth;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ValueOrigin &vo) {
  vo.print(os);
  return os;
}

namespace llvm {
template <> struct DenseMapInfo<ValueOrigin> {
  using PosType = decltype(std::declval<ValueOrigin>().position);
  static constexpr PosType impossiblePosition =
      std::numeric_limits<PosType>::min();

  static ValueOrigin getEmptyKey() {
    return ValueOrigin(impossiblePosition, -1);
  }

  static ValueOrigin getTombstoneKey() {
    return ValueOrigin(impossiblePosition, -2);
  }

  static unsigned getHashValue(ValueOrigin value) {
    return llvm::hash_combine(llvm::hash_value(value.position),
                              llvm::hash_value(value.depth));
  }

  static bool isEqual(const ValueOrigin &lhs, const ValueOrigin &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

// --------------------------------

class ValueOrigins {
public:
  enum class State {
    Undefined, ///< Has not been analyzed yet (lattice bottom).
    Defined,   ///< Has specific positions.
    Unknown    ///< Analyzed and may be any origin (lattice top).
  };

  ValueOrigins() : state(State::Undefined) {}

  explicit ValueOrigins(ValueOrigin origin) : state(State::Defined) {
    origins.insert(origin);
  }

  static const ValueOrigins &getUnknown() {
    static const ValueOrigins unknown(State::Unknown);
    return unknown;
  }

  static const ValueOrigins &getUndefined() {
    static const ValueOrigins undefined(State::Undefined);
    return undefined;
  }

  bool isDefined() const { return state == State::Defined; }
  bool isUndefined() const { return state == State::Undefined; }
  bool isUnknown() const { return state == State::Unknown; }

  ChangeResult join(const ValueOrigins &other);

  bool operator==(const ValueOrigins &other) const {
    return state == other.state && origins == other.origins;
  }

  static ValueOrigins getLoaded(const ValueOrigins &other);
  static ValueOrigins getStored(const ValueOrigins &other);

  Attribute getAsAttribute(MLIRContext *context) const;

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump();

private:
  ValueOrigins(State state) : state(state) {}

  void changeDepth(int32_t add);

  /// Potential origins of a value. Keep as a sorted std::set for now to ensure
  /// deterministic everything.
  llvm::SmallDenseSet<ValueOrigin> origins;

  State state;
};

void ValueOrigins::dump() { return print(llvm::errs()); }

Attribute ValueOrigins::getAsAttribute(MLIRContext *context) const {
  if (!isDefined()) {
    std::string str;
    llvm::raw_string_ostream os(str);
    print(os);
    return StringAttr::get(context, os.str());
  }

  SmallVector<ValueOrigin> sorted(origins.begin(), origins.end());
  llvm::sort(sorted);
  auto attrs =
      llvm::map_to_vector(sorted, [context](ValueOrigin vo) -> Attribute {
        return DenseI32ArrayAttr::get(context, {vo.position, vo.depth});
      });
  return ArrayAttr::get(context, attrs);
}

void ValueOrigins::changeDepth(int32_t add) {
  for (ValueOrigin &o : origins)
    o.depth += add;
}

ValueOrigins ValueOrigins::getLoaded(const ValueOrigins &other) {
  ValueOrigins result(other);
  result.changeDepth(-1);
  return result;
}

ValueOrigins ValueOrigins::getStored(const ValueOrigins &other) {
  ValueOrigins result(other);
  result.changeDepth(+1);
  return result;
}

ChangeResult ValueOrigins::join(const ValueOrigins &other) {
  if (isUnknown())
    return ChangeResult::NoChange;
  if (other.isUnknown()) {
    state = State::Unknown;
    origins.clear();
    return ChangeResult::Change;
  }
  if (isUndefined() && other.isUndefined())
    return ChangeResult::NoChange;

  state = State::Defined;
  return llvm::set_union(origins, other.origins) ? ChangeResult::Change
                                                 : ChangeResult::NoChange;
}

void ValueOrigins::print(llvm::raw_ostream &os) const {
  if (isUndefined()) {
    os << "<undefined>";
    return;
  }
  if (isUnknown()) {
    os << "<unknown>";
    return;
  }

  os << "{";
  SmallVector<ValueOrigin> sorted(origins.begin(), origins.end());
  llvm::sort(sorted);
  llvm::interleaveComma(sorted, os);
  os << "}";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ValueOrigins &av) {
  av.print(os);
  return os;
}

// --------------------------------

class ValueOriginsLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  ValueOriginsLattice(Value point, ValueOrigins &&origins)
      : AbstractSparseLattice(point), origins(std::move(origins)) {}

  ChangeResult join(const AbstractSparseLattice &other) override {
    return origins.join(
        static_cast<const ValueOriginsLattice &>(other).origins);
  }
  ChangeResult join(const ValueOrigins &other) { return origins.join(other); }

  ChangeResult markUnknown() {
    return origins.join(ValueOrigins::getUnknown());
  }

  void print(llvm::raw_ostream &os) const override {
    os << "origins: " << origins;
  }

  Attribute getAsAttribute(MLIRContext *context) const;

private:
  ValueOrigins origins;
};

// -------------------------------------------

class ValueOriginsDenseLattice : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &other) override;
  ChangeResult join(DistinctAttr valueClass, ValueOrigins origin) {
    if (origin.isUndefined())
      return ChangeResult::NoChange;
    return origins[valueClass].join(origin);
  }

  ChangeResult meet(const AbstractDenseLattice &other) override {
    return ChangeResult::NoChange;
  }

  // MLIR backward analysis misuses terminology...
  // ChangeResult meet(const AbstractDenseLattice &other) override {
  //   return join(other);
  // }

  ChangeResult markAllUnknown();

  const ValueOrigins &getOriginsOf(DistinctAttr valueClass) const {
    auto it = origins.find(valueClass);
    if (it == origins.end())
      return ValueOrigins::getUndefined();
    return it->second;
  }

  void print(llvm::raw_ostream &os) const override;

  Attribute getAsAttribute(MLIRContext *context) const;

private:
  // TODO: we don't necessarily need a vector here if we fix the printing.
  llvm::MapVector<DistinctAttr, ValueOrigins> origins;
  // TODO(zinenko): combined lattice: map for Value + map for DistinctAttr?
  // TODO(zinenko): alt. create DistinctAttr for all values?
};

Attribute ValueOriginsDenseLattice::getAsAttribute(MLIRContext *context) const {
  SmallVector<Attribute> entries;
  entries.reserve(origins.size());
  for (auto &&[valueClass, origin] : origins) {
    entries.emplace_back(
        ArrayAttr::get(context, {valueClass, origin.getAsAttribute(context)}));
  }
  return ArrayAttr::get(context, entries);
}

ChangeResult ValueOriginsDenseLattice::join(const AbstractDenseLattice &other) {
  const auto &rhs = static_cast<const ValueOriginsDenseLattice &>(other);
  ChangeResult change = ChangeResult::NoChange;
  // Note that `operator[]` on the map automatically injects a pair into the map
  // where the value is default-constructed to be in the "undefined" state.
  // Joining with such a value is equivalent to copying it over from the RHS.
  for (auto &&[value, valueOrigins] : rhs.origins)
    change |= origins[value].join(valueOrigins);
  return change;
}

ChangeResult ValueOriginsDenseLattice::markAllUnknown() {
  ChangeResult change = ChangeResult::NoChange;
  for (auto &valueOrigins : llvm::make_second_range(origins))
    change |= valueOrigins.join(ValueOrigins::getUnknown());
  return change;
}

void ValueOriginsDenseLattice::print(llvm::raw_ostream &os) const {
  os << "{";
  std::optional<AsmState> state;
  for (auto &&[valueClass, valueOrigins] : origins) {
    if (!state)
      state.emplace(valueClass.getContext());
    valueClass.print(os << "\n  ", *state);
    valueOrigins.print(os << " -> ");
  }
  if (!origins.empty())
    os << "\n";
  os << "}";
}

// ---------------------------------------

class DenseForwardActivityAnnotationAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<ValueOriginsDenseLattice> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, const ValueOriginsDenseLattice &before,
                      ValueOriginsDenseLattice *after) override;

  // This is actually the pessimistic fixpoint, and does _not_ seem to be used
  // as the actual entry state.
  void setToEntryState(ValueOriginsDenseLattice *lattice) override;

  DistinctAttr getFixedOriginalClass(Value value) const {
    return originalClasses.getFixedOriginalClass(value);
  }

  // TODO(zinenko): consider multiple functions
  FunctionOpInterface entryPoint;

private:
  ChangeResult processReadEffect(Value value, Operation *op,
                                 const ValueOriginsDenseLattice &before,
                                 ValueOriginsDenseLattice *after);
  ChangeResult processWriteEffect(Value value, Operation *op,
                                  const ValueOriginsDenseLattice &before,
                                  ValueOriginsDenseLattice *after);

  enzyme::AliasClassSet allValueClassesFor(ProgramPoint point, Value value);

  enzyme::OriginalClasses originalClasses;
};

// TODO: make this an interface or another configuration (maybe we want to
// differentiate wrt integers for some reason, e.g. extreme quantization).
static bool mayVariate(Value v) {
  return isa<FloatType, ComplexType>(v.getType());
}

static bool isMemoryLike(Value v) {
  return isa<LLVM::LLVMPointerType, MemRefType>(v.getType());
}

void DenseForwardActivityAnnotationAnalysis::setToEntryState(
    ValueOriginsDenseLattice *lattice) {
  // TODO(zinenko): we cannot actually do this because this function gets called
  // for the entry block of a function when we don't know all call sites of the
  // function.
  //
  // Instead, we would like to call some initialization function there.
  //
  // We also cannot avoid doing this in the other cases, e.g., where we don't
  // understand regular control flow.
  //
  // Should we allow the user to override this pessimistic behavior by
  // implementing the visitCallControlFlowTransfer hook for call == nullptr?
  //
  // But if at this point, we haven't yet added the values to the lattice, they
  // are implicit-undefined. So we _won't_ mark them unknown (and this is not
  // the pessimistic fixpoint / top). We will only pessimize the values we have
  // previously seen.
  // TODO: this may involve values in the caller.
  // TODO: is this a happy side effect of our modeling? should we actually
  // mark unknown all values, even the future ones?
  lattice->markAllUnknown();

  // TODO(zinenko): same should happen in alias analysis
  SmallVector<Value> aliasingArguments;
  for (BlockArgument arg :
       entryPoint.getFunctionBody().front().getArguments()) {
    if (!isMemoryLike(arg))
      continue;

    if (entryPoint.getArgAttr(arg.getArgNumber(),
                              LLVM::LLVMDialect::getNoAliasAttrName())) {
      continue;
    }

    aliasingArguments.push_back(arg);
  }
  DistinctAttr commonAliasClass;
  if (aliasingArguments.size() != 0) {
    commonAliasClass = originalClasses.getSameOriginalClass(aliasingArguments,
                                                            "func-arg-common");
  }

  // But we still somehow need to initialize the state for function args.
  // TODO(zinenko): this is still hacky.
  ChangeResult changed = ChangeResult::NoChange;
  for (BlockArgument arg :
       entryPoint.getFunctionBody().front().getArguments()) {
    if (!mayVariate(arg) && !isMemoryLike(arg))
      continue;

    std::string debugLabel = "func-arg-" + std::to_string(arg.getArgNumber());
    DistinctAttr thisAliasClass =
        llvm::is_contained(aliasingArguments, arg)
            ? commonAliasClass
            : originalClasses.getOriginalClass(arg, debugLabel);

    // We cannot actually "initialize" or "set" here because we could have
    // updated the function arguments to have other origins because of stores,
    // and we can't (and don't want to) roll that back.
    changed |= lattice->join(thisAliasClass,
                             ValueOrigins(ValueOrigin(arg.getArgNumber(), 0)));
  }
  propagateIfChanged(lattice, changed);
}

enzyme::AliasClassSet
DenseForwardActivityAnnotationAnalysis::allValueClassesFor(
    ProgramPoint dependent, Value value) {
  enzyme::AliasClassSet valueClasses(originalClasses.getOriginalClass(value));
  if (isMemoryLike(value))
    (void)valueClasses.join(
        getOrCreateFor<enzyme::AliasClassLattice>(dependent, value)
            ->getAliasClassesObject());
  return valueClasses;
}

static ChangeResult transferActivityCross(
    const enzyme::AliasClassSet &destClasses,
    const enzyme::AliasClassSet &sourceClasses,
    const ValueOriginsDenseLattice &before, ValueOriginsDenseLattice *after,
    function_ref<ChangeResult(DistinctAttr, DistinctAttr)> join) {
  return destClasses.foreachClass([&](DistinctAttr destClass,
                                      enzyme::AliasClassSet::State state) {
    if (state == enzyme::AliasClassSet::State::Undefined)
      return ChangeResult::NoChange;
    if (state == enzyme::AliasClassSet::State::Unknown)
      return after->markAllUnknown();

    return sourceClasses.foreachClass(
        [&](DistinctAttr sourceClasses, enzyme::AliasClassSet::State state) {
          if (state == enzyme::AliasClassSet::State::Undefined)
            return ChangeResult::NoChange;
          if (state == enzyme::AliasClassSet::State::Unknown)
            return after->join(destClass, ValueOrigins::getUnknown());
          return join(destClass, sourceClasses);
        });
  });
}

void DenseForwardActivityAnnotationAnalysis::visitOperation(
    Operation *op, const ValueOriginsDenseLattice &before,
    ValueOriginsDenseLattice *after) {
  ChangeResult change = after->join(before);
  auto scope =
      llvm::make_scope_exit([&] { propagateIfChanged(after, change); });

  // Direct transfer by cross-product of origins.
  // TODO: interface for activity transfer.
  if (isPure(op)) {
    enzyme::AliasClassSet operandValueClasses;
    enzyme::AliasClassSet resultValueClasses;

    for (Value result : op->getResults()) {
      (void)resultValueClasses.join(allValueClassesFor(op, result));
    }
    for (Value operand : op->getOperands()) {
      (void)operandValueClasses.join(allValueClassesFor(op, operand));
    }

    change |= transferActivityCross(
        resultValueClasses, operandValueClasses, before, after,
        [&](DistinctAttr resultClass, DistinctAttr operandClass) {
          return after->join(resultClass, before.getOriginsOf(operandClass));
        });
    return;
  }

  // Pointer based:
  // if loading a value:
  //  - get alias classes of said value
  //  - get _origins_ it may be pointing to (NEEDED)
  // need a points-to analysis for values that are not only pointers, but also
  // variables

  // are we getting in situation where we need an "alias" analysis for potential
  // origins (implicit assumption that all arguments are "noalias" for this
  // purpose)
  // + a points-to-"pointer" analysis for classes? Looks like it,
  // but we also need true alias classes

  auto memEffectsIface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectsIface)
    return propagateIfChanged(after, after->markAllUnknown());

  SmallVector<MemoryEffects::EffectInstance> effects;
  memEffectsIface.getEffects(effects);
  ChangeResult needsPropagation = ChangeResult::NoChange;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    // TODO(zinenko): is this too conservative?
    if (!effect.getValue())
      return propagateIfChanged(after, after->markAllUnknown());

    if (!isMemoryLike(effect.getValue()))
      continue;

    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      needsPropagation |=
          processReadEffect(effect.getValue(), op, before, after);
    }

    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      needsPropagation |=
          processWriteEffect(effect.getValue(), op, before, after);
    }
  }
  propagateIfChanged(after, needsPropagation);
}

ChangeResult DenseForwardActivityAnnotationAnalysis::processReadEffect(
    Value value, Operation *op, const ValueOriginsDenseLattice &before,
    ValueOriginsDenseLattice *after) {
  enzyme::AliasClassSet resultClasses;
  for (Value result : op->getResults())
    (void)resultClasses.join(allValueClassesFor(op, result));

  enzyme::AliasClassSet operandClasses = allValueClassesFor(op, value);

  return transferActivityCross(
      resultClasses, operandClasses, before, after,
      [&](DistinctAttr resultClass, DistinctAttr operandClass) {
        return after->join(resultClass, ValueOrigins::getLoaded(
                                            before.getOriginsOf(operandClass)));
      });
}

// TODO: turn into interface
static Value getStoredValue(Operation *op) {
  if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
    return store.getValue();
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    return store.getValueToStore();
  }
  return nullptr;
}

ChangeResult DenseForwardActivityAnnotationAnalysis::processWriteEffect(
    Value value, Operation *op, const ValueOriginsDenseLattice &before,
    ValueOriginsDenseLattice *after) {
  Value stored = getStoredValue(op);
  enzyme::AliasClassSet storedClasses =
      stored ? allValueClassesFor(op, stored)
             : enzyme::AliasClassSet::getUnknown();

  enzyme::AliasClassSet destClasses = allValueClassesFor(op, value);

  return transferActivityCross(
      destClasses, storedClasses, before, after,
      [&](DistinctAttr resultClass, DistinctAttr storedClass) {
        return after->join(resultClass, ValueOrigins::getStored(
                                            before.getOriginsOf(storedClass)));
      });
}

// --------------------------------------

static LogicalResult runActivityAnnotationDebug(FunctionOpInterface function) {
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<enzyme::AliasAnalysis>(function->getContext());
  solver.load<enzyme::PointsToPointerAnalysis>();

  auto *analysis = solver.load<DenseForwardActivityAnnotationAnalysis>();
  analysis->entryPoint = function;
  analysis->setToEntryState(solver.getOrCreateState<ValueOriginsDenseLattice>(
      &function.getFunctionBody().front()));

  // auto *ptoa = solver.load<PointsToOriginAnalysis>();
  // ptoa->entryPoint = function;
  // ptoa->aliasAnalysis = aa;
  // ptoa->setToEntryState(solver.getOrCreateState<PointsToOriginsDenseLattice>(
  //     &function.getFunctionBody().front()));

  if (failed(solver.initializeAndRun(function)))
    return function->emitError() << "couldn't run activity annotation";

  auto getValueClassesAttr =
      [&](Value value, const ValueOriginsDenseLattice &origins) -> Attribute {
    enzyme::AliasClassSet aliasClasses;
    if (const auto *lattice =
            solver.getOrCreateState<enzyme::AliasClassLattice>(value)) {
      aliasClasses.join(lattice->getAliasClassesObject());
    }
    aliasClasses.join(
        enzyme::AliasClassSet(analysis->getFixedOriginalClass(value)));

    // TODO(zinenko): lift to the class itself.
    SmallVector<Attribute> attrAliasClasses;
    aliasClasses.foreachClass(
        [&](DistinctAttr aliasClass, enzyme::AliasClassSet::State state) {
          if (state == enzyme::AliasClassSet::State::Undefined) {
            attrAliasClasses.push_back(
                StringAttr::get(value.getContext(), "<undefined>"));
          } else if (state == enzyme::AliasClassSet::State::Unknown) {
            attrAliasClasses.push_back(
                StringAttr::get(value.getContext(), "<unknown>"));
          } else {
            attrAliasClasses.push_back(aliasClass);
          }
          return ChangeResult::NoChange;
        });

    if (attrAliasClasses.size() == 1 &&
        isa<StringAttr>(attrAliasClasses.front()))
      return attrAliasClasses.front();
    return ArrayAttr::get(value.getContext(), attrAliasClasses);
  };

  MLIRContext *ctx = function.getContext();
  function.walk([&](Operation *op) {
    // First op in the block.
    if (!op->getPrevNode()) {
      const auto *origins =
          solver.getOrCreateState<ValueOriginsDenseLattice>(op->getBlock());

      op->setAttr("enzyme.activity_annotation.origin.before",
                  origins ? origins->getAsAttribute(ctx) : UnitAttr::get(ctx));

      auto argAttrs = llvm::map_to_vector(
          op->getBlock()->getArguments(), [&](Value operand) {
            return getValueClassesAttr(operand, *origins);
          });
      op->setAttr("enzyme.activity_annotation.classes.block",
                  ArrayAttr::get(ctx, argAttrs));
    }
    const auto *origins = solver.getOrCreateState<ValueOriginsDenseLattice>(op);
    op->setAttr("enzyme.activity_annotation.origin.after",
                origins ? origins->getAsAttribute(ctx) : UnitAttr::get(ctx));
    auto operandAttrs =
        llvm::map_to_vector(op->getOperands(), [&](Value operand) {
          return getValueClassesAttr(operand, *origins);
        });
    op->setAttr("enzyme.activity_annotation.classes.operands",
                ArrayAttr::get(ctx, operandAttrs));
    auto resultAttrs =
        llvm::map_to_vector(op->getResults(), [&](Value operand) {
          return getValueClassesAttr(operand, *origins);
        });
    op->setAttr("enzyme.activity_annotation.classes.results",
                ArrayAttr::get(ctx, resultAttrs));
  });
  return success();
}

struct ActivityAnnotationDebugPass
    : public PassWrapper<ActivityAnnotationDebugPass,
                         InterfacePass<FunctionOpInterface>> {
  StringRef getArgument() const override {
    return "enzyme-activity-annotation-debug";
  }

  void runOnOperation() override {
    if (failed(runActivityAnnotationDebug(getOperation())))
      return signalPassFailure();
  }
};

void registerActivityAnnotationDebugPass() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<ActivityAnnotationDebugPass>();
  });
}
