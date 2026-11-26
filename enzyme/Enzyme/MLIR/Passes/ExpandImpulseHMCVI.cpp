//===- ExpandImpulseHMCVI.cpp - Lower HMC, NUTS and VI inference ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers `impulse.infer` and `impulse.infer_region` with an HMC or NUTS config
// to warmup and sampling loops per chain, and with a VI config to an SVI loop.
//
//===----------------------------------------------------------------------===//

#include "ExpandImpulse.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Impulse/InferStateLayout.h"
#include "Dialect/Ops.h"
#include "Interfaces/HMCUtils.h"
#include "Interfaces/ImpulseUtils.h"
#include "Interfaces/VIUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/APFloat.h"

#include <type_traits>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;

/// Fails after emitting on `op` when `support`, the support of the site at
/// `address`, lacks a bound or the event size that unconstraining reads.
static LogicalResult verifyUnconstrainable(Operation *op, Attribute address,
                                           impulse::SupportAttr support) {
  if (!support)
    return success();
  impulse::SupportKind kind = support.getKind();
  SmallVector<StringRef> missing;
  if ((kind == impulse::SupportKind::INTERVAL ||
       kind == impulse::SupportKind::GREATER_THAN) &&
      !support.getLowerBound())
    missing.push_back("`lower_bound`");
  if ((kind == impulse::SupportKind::INTERVAL ||
       kind == impulse::SupportKind::LESS_THAN) &&
      !support.getUpperBound())
    missing.push_back("`upper_bound`");
  if (kind == impulse::SupportKind::SIMPLEX && !support.getEventSize())
    missing.push_back("`event_size`");
  if (missing.empty())
    return success();
  return op->emitError("Impulse: the ")
         << impulse::stringifySupportKind(kind) << " support of the site at "
         << address << " has no " << llvm::join(missing, " and ")
         << ", which unconstraining the site needs";
}

/// Checks the supports of the sites at the top-level addresses of `layout`.
static LogicalResult verifySelectedSupports(Operation *op,
                                            const FlatTraceLayout &layout) {
  for (const FlatTraceLayout::Entry &entry : layout.entries) {
    if (entry.address.size() != 1)
      continue;
    impulse::SampleOp site = layout.siteBySymbol.lookup(entry.address[0]);
    if (failed(verifyUnconstrainable(op, entry.address, site.getSupportAttr())))
      return failure();
  }
  return success();
}

/// Checks the supports of the sites of `regionOp` at the addresses in its
/// `selection`.
static LogicalResult verifySelectedSupports(impulse::InferRegionOp regionOp) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  DenseSet<Attribute> selected(selection.begin(), selection.end());
  WalkResult result =
      regionOp.getSampling().walk([&](impulse::SampleRegionOp site) {
        Attribute address = site.getAddressAttr();
        if (address && selected.contains(address) &&
            failed(verifyUnconstrainable(regionOp, address,
                                         site.getSupportAttr())))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  return failure(result.wasInterrupted());
}

/// The selected addresses in `all_addresses` order if
/// `positionDeclarationOrder`, else in `selection` order, each mapped through
/// `key`; an address that `key` maps to null is skipped.
static SmallVector<Attribute>
orderSelected(ArrayAttr selection, ArrayAttr allAddresses,
              bool positionDeclarationOrder,
              function_ref<Attribute(Attribute)> key) {
  SmallVector<Attribute> ordered;
  if (!positionDeclarationOrder) {
    for (Attribute address : selection)
      if (Attribute selected = key(address))
        ordered.push_back(selected);
    return ordered;
  }
  llvm::SmallDenseSet<Attribute, 4> selected;
  for (Attribute address : selection)
    if (Attribute keyed = key(address))
      selected.insert(keyed);
  for (Attribute address : allAddresses) {
    Attribute keyed = key(address);
    if (keyed && selected.contains(keyed))
      ordered.push_back(keyed);
  }
  return ordered;
}

/// Appends to `supports` a site of `size` constrained values on `support` at
/// trace column `traceOffset`, placed in the position after the sites already
/// in `supports`.
static void appendSupport(SmallVectorImpl<impulse::SupportInfo> &supports,
                          int64_t traceOffset, int64_t size,
                          impulse::SupportAttr support) {
  int64_t constrainedOffset = 0;
  int64_t unconstrainedOffset = 0;
  if (!supports.empty()) {
    const impulse::SupportInfo &last = supports.back();
    constrainedOffset = last.constrainedOffset + last.constrainedSize;
    unconstrainedOffset = last.unconstrainedOffset + last.unconstrainedSize;
  }
  int64_t unconstrainedSize =
      support ? support.getUnconstrainedSize(size) : size;
  supports.emplace_back(constrainedOffset, traceOffset, size,
                        unconstrainedOffset, unconstrainedSize, support);
}

/// Support info for the top-level addresses in `selection`, whose sites
/// `selected` lays out. Positions follow `all_addresses` order when
/// `positionDeclarationOrder` is set, else `selection` order; trace offsets
/// follow `allAddresses`.
static SmallVector<impulse::SupportInfo> collectSupportInfoForSelection(
    Operation *op, FunctionOpInterface fn, ArrayAttr selection,
    const FlatTraceLayout &selected, ArrayAttr allAddresses,
    SymbolTableCollection &symbolTable, bool positionDeclarationOrder) {
  DenseMap<Attribute, int64_t> sizeOf;
  for (const FlatTraceLayout::Entry &entry : selected.entries)
    if (entry.address.size() == 1)
      sizeOf.try_emplace(entry.address[0], entry.size);
  auto topLevelSymbol = [](Attribute address) -> Attribute {
    auto path = cast<ArrayAttr>(address);
    return path.size() == 1 ? path[0] : Attribute();
  };
  SmallVector<impulse::SupportInfo> supports;
  for (Attribute symbol : orderSelected(
           selection, allAddresses, positionDeclarationOrder, topLevelSymbol)) {
    // Only the addresses before the site must resolve here; the others are
    // checked where the model is traced.
    int64_t traceOffset = computeOffsetForSampleInSelection(
        op, fn, allAddresses, symbol, symbolTable);
    if (traceOffset < 0) {
      op->emitError("Symbol in selection not found in all_addresses - cannot "
                    "determine trace offset for scattered selection");
      return {};
    }
    appendSupport(supports, traceOffset, sizeOf.lookup(symbol),
                  selected.siteBySymbol.lookup(symbol).getSupportAttr());
  }
  return supports;
}

namespace {
/// The position layout of the selected sites of a region-form infer op.
struct RegionPositionLayout {
  SmallVector<impulse::SupportInfo> supports;
  /// Position column of each value of the selected sites, in `selection`
  /// order.
  SmallVector<int64_t> argOffsets;
};
} // namespace

/// Lays out the selected sites of `regionOp` in `all_addresses` order if
/// `positionDeclarationOrder`, else in `selection` order; trace offsets follow
/// `all_addresses`. Fails when an address has no site in the sampler.
static FailureOr<RegionPositionLayout>
collectRegionLayout(impulse::InferRegionOp regionOp,
                    bool positionDeclarationOrder) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || !allAddresses || regionOp.getSampling().empty())
    return failure();
  // Branch-local sites also hold trace columns, so walk the whole sampler.
  FailureOr<RegionTraceLayout> trace =
      getRegionTraceLayout(regionOp, /*topLevelOnly=*/false);
  if (failed(trace))
    return failure();

  RegionPositionLayout layout;
  DenseMap<Attribute, int64_t> positionColumn;
  for (Attribute address :
       orderSelected(selection, allAddresses, positionDeclarationOrder,
                     [](Attribute address) { return address; })) {
    impulse::SampleRegionOp site = trace->siteByAddress.lookup(address);
    if (!site || !trace->columnOf.count(address))
      return failure();
    appendSupport(layout.supports, trace->columnOf.lookup(address),
                  sampleRegionSize(site), site.getSupportAttr());
    positionColumn[address] = layout.supports.back().constrainedOffset;
  }

  for (Attribute address : selection) {
    impulse::SampleRegionOp site = trace->siteByAddress.lookup(address);
    if (!site)
      return failure();
    int64_t column = positionColumn.lookup(address);
    for (Value result : llvm::drop_begin(site->getResults())) {
      layout.argOffsets.push_back(column);
      column += cast<RankedTensorType>(result.getType()).getNumElements();
    }
  }
  return layout;
}

namespace {
/// HMC/NUTS warmup adaptation state, in `state_schema` slot order from
/// `da_log_step_size` through `window_idx`. `welford` is unset while the warmup
/// does not adapt the mass matrix.
struct AdaptationState {
  DualAveragingState dualAveraging;
  std::optional<WelfordState> welford;
  Value windowIdx;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = dualAveraging.toValues();
    if (welford)
      llvm::append_range(values, welford->toValues());
    values.push_back(windowIdx);
    return values;
  }
  SmallVector<Type> getTypes() const {
    return llvm::to_vector(ValueRange(toValues()).getTypes());
  }
  /// Reads the state, with Welford accumulators if `hasWelford`, from the
  /// front of `values` and drops what it read.
  static AdaptationState consume(ArrayRef<Value> &values, bool hasWelford) {
    AdaptationState state;
    state.dualAveraging = DualAveragingState::fromValues(
        values.take_front(DualAveragingState::kNumValues));
    values = values.drop_front(DualAveragingState::kNumValues);
    if (hasWelford) {
      state.welford =
          WelfordState::fromValues(values.take_front(WelfordState::kNumValues));
      values = values.drop_front(WelfordState::kNumValues);
    }
    state.windowIdx = values.front();
    values = values.drop_front();
    return state;
  }
};

/// Values carried by the HMC/NUTS warmup loop.
struct WarmupCarry {
  ChainPoint chain;
  Value stepSize;
  Value invMass;
  Value massMatrixSqrt;
  AdaptationState adaptation;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = {
        chain.q,  chain.grad, chain.potentialEnergy, chain.rng,
        stepSize, invMass,    massMatrixSqrt};
    llvm::append_range(values, adaptation.toValues());
    return values;
  }
  /// The position, gradient, potential energy and step size are typed by
  /// `positionType` and `scalarType`.
  SmallVector<Type> getTypes(RankedTensorType positionType,
                             RankedTensorType scalarType) const {
    SmallVector<Type> types = {positionType,
                               positionType,
                               scalarType,
                               chain.rng.getType(),
                               scalarType,
                               invMass.getType(),
                               massMatrixSqrt.getType()};
    llvm::append_range(types, adaptation.getTypes());
    return types;
  }
  static WarmupCarry fromValues(ValueRange range, bool hasWelford) {
    SmallVector<Value> storage = llvm::to_vector(range);
    ArrayRef<Value> values = storage;
    WarmupCarry carry;
    carry.chain = {values[0], values[1], values[2], values[3]};
    carry.stepSize = values[4];
    carry.invMass = values[5];
    carry.massMatrixSqrt = values[6];
    values = values.drop_front(7);
    carry.adaptation = AdaptationState::consume(values, hasWelford);
    return carry;
  }
};

/// The results of one HMC/NUTS chain, in the order of the op's results.
struct ChainResult {
  Value trace;
  Value diagnostics;
  Value logDensities;
  Value rng;
  SmallVector<Value> stateOut;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = {trace, diagnostics, logDensities, rng};
    llvm::append_range(values, stateOut);
    return values;
  }
};

/// Emits the HMC/NUTS chains of one `impulse.infer` or `impulse.infer_region`.
/// The constructor reads what all chains share and creates the kernel's
/// constant operands; `emitChains` then emits the chains one after another.
class MCMCChainEmitter {
public:
  template <typename OpT>
  MCMCChainEmitter(OpBuilder &builder, OpT op, ModelContext model);

  /// Emits every chain. Several chains each start from their own key: the
  /// input key split once per chain, or on resume the keys saved in the `rng`
  /// slot.
  SmallVector<ChainResult> emitChains(Value rng);

private:
  /// Constants of the warmup adaptation schedule.
  struct WarmupSchedule {
    Value zero;
    Value one;
    SmallVector<Value> windowEnds;
    Value lastWindow;
    Value lastIteration;
  };

  /// Emits chain `chainIdx`, which draws from key `rng`.
  ChainResult emitChain(Value rng, int64_t chainIdx);

  /// The `state_in` value of slot `name`, or null on a cold start.
  Value stateInSlot(StringRef name) const;
  /// Like `stateInSlot`, but only chain `chainIdx` of a chain-major slot.
  Value chainStateInSlot(StringRef name, int64_t chainIdx);

  /// The HMC context of `model` with the given mass matrix and step size.
  HMCContext makeHMCContext(Value invMass, Value massMatrixSqrt,
                            Value stepSize) const;
  /// The NUTS context of `model` with the given mass matrix and step size,
  /// for a transition from potential energy `potentialEnergy`.
  NUTSContext makeNUTSContext(Value invMass, Value massMatrixSqrt,
                              Value stepSize, Value potentialEnergy) const;
  /// One HMC or NUTS transition from `state`.
  MCMCKernelResult runKernel(const ChainPoint &state, Value stepSize,
                             Value invMass, Value massMatrixSqrt);

  /// The chain's resumed state, or the initial state computed by InitHMC.
  ChainPoint initChainState(Value rng, int64_t chainIdx, Value stepSize,
                            Value invMass, Value massMatrixSqrt);
  /// Returns `saved` when set, else a fresh adaptation state for `stepSize`;
  /// a fresh state gets Welford accumulators only if `withWelford`.
  AdaptationState
  loadOrInitAdaptation(Value stepSize, bool withWelford,
                       const std::optional<AdaptationState> &saved);
  /// Emits the warmup loop and returns its results.
  WarmupCarry emitWarmup(const ChainPoint &start, Value stepSize, Value invMass,
                         Value massMatrixSqrt,
                         const std::optional<AdaptationState> &saved,
                         int64_t chainIdx);
  /// Emits one warmup iteration at global iteration `iteration` and returns
  /// the values it carries to the next one.
  WarmupCarry emitWarmupStep(const WarmupCarry &carry, Value iteration,
                             const WarmupSchedule &schedule);
  /// Emits the end of a middle adaptation window: when `atMiddleWindowEnd`
  /// holds, the mass matrix is re-estimated from the Welford accumulators of
  /// `next` and the adapted accumulators restart. Updates `next` to the
  /// selected values.
  void emitWindowEndRestart(WarmupCarry &next, Value atMiddleWindowEnd);
  /// Emits the sampling loop from `state`, leaves the chain's final state in
  /// `state` and returns the collected samples and diagnostics.
  ChainResult emitSampling(ChainPoint &state, Value stepSize, Value invMass,
                           Value massMatrixSqrt);
  /// The chain's `state_out` values in the order of the canonical
  /// `state_schema`, which the op verifier enforces: the chain state, the
  /// step size and the inverse mass matrix, then with `adaptation` its values
  /// and the warmup offset.
  SmallVector<Value>
  packStateOut(const ChainPoint &state, Value stepSize, Value invMass,
               const std::optional<AdaptationState> &adaptation,
               int64_t chainIdx);

  OpBuilder &builder;
  Location loc;
  ModelContext model;

  int64_t numWarmup;
  int64_t totalWarmup;
  int64_t numSamples;
  int64_t thinning;
  int64_t numChains;

  ArrayAttr schema;
  ValueRange stateIn;
  Value initPosition;
  bool exposeAdaptation;
  std::optional<AdaptationState> savedAdaptation;
  /// Step size and inverse mass matrix every chain starts from, unless a
  /// chain-major `state_in` slot overrides them per chain.
  Value initStepSize;
  Value initInvMass;
  /// Whether the inverse mass matrix and its Welford estimate are diagonal.
  bool diagonal = true;
  WelfordConfig welfordConfig;

  bool isHMC;
  bool isNUTS;
  bool adaptStepSize = false;
  bool adaptMassMatrix = false;
  Value trajectoryLength;
  int64_t numSteps = 0;
  Value maxDeltaEnergy;
  int64_t maxTreeDepth = 0;

  Type elemType;
  RankedTensorType positionType;
  RankedTensorType scalarType;
  RankedTensorType i64TensorType;
};
} // namespace

/// Stacks the per-chain results into the op's chain-major results. The output
/// RNG state is the last chain's key; with a state schema, the keys of all
/// chains also go into the trailing `rng` slot.
static SmallVector<Value> stackChainResults(OpBuilder &builder, Location loc,
                                            ArrayRef<ChainResult> chains,
                                            bool hasStateSchema) {
  Value zeroIdx = emitI64ScalarConstant(builder, loc, 0);
  int64_t numChains = chains.size();

  auto stack = [&](function_ref<Value(const ChainResult &)> pick,
                   bool isRng) -> Value {
    auto type = cast<RankedTensorType>(pick(chains.front()).getType());
    SmallVector<int64_t> stackedShape = {numChains};
    llvm::append_range(stackedShape, type.getShape());
    auto stackedType =
        RankedTensorType::get(stackedShape, type.getElementType());
    SmallVector<int64_t> oneShape = {1};
    llvm::append_range(oneShape, type.getShape());
    auto oneType = RankedTensorType::get(oneShape, type.getElementType());
    Value buffer;
    // ui64 has no arith.constant, so the key buffer starts from chain 0's key.
    if (isRng)
      buffer = enzyme::BroadcastOp::create(
          builder, loc, stackedType, pick(chains.front()),
          builder.getDenseI64ArrayAttr({numChains}));
    else
      buffer = arith::ConstantOp::create(builder, loc, stackedType,
                                         builder.getZeroAttr(stackedType));
    for (auto [chainIdx, chain] : llvm::enumerate(chains)) {
      Value reshaped =
          impulse::ReshapeOp::create(builder, loc, oneType, pick(chain));
      SmallVector<Value> indices = {
          emitI64ScalarConstant(builder, loc, chainIdx)};
      indices.append(type.getRank(), zeroIdx);
      buffer = impulse::DynamicUpdateSliceOp::create(builder, loc, stackedType,
                                                     buffer, reshaped, indices);
    }
    return buffer;
  };

  SmallVector<Value> results;
  results.push_back(stack([](const ChainResult &chain) { return chain.trace; },
                          /*isRng=*/false));
  results.push_back(
      stack([](const ChainResult &chain) { return chain.diagnostics; },
            /*isRng=*/false));
  results.push_back(
      stack([](const ChainResult &chain) { return chain.logDensities; },
            /*isRng=*/false));
  results.push_back(chains.back().rng);
  for (size_t i = 0, e = chains.front().stateOut.size(); i < e; ++i)
    results.push_back(
        stack([i](const ChainResult &chain) { return chain.stateOut[i]; },
              /*isRng=*/false));
  if (hasStateSchema)
    results.push_back(stack([](const ChainResult &chain) { return chain.rng; },
                            /*isRng=*/true));
  return results;
}

/// Reads the operands that the HMC, NUTS and VI lowerings share; the caller
/// lays out the position.
template <typename OpT>
static ModelContext readInferModel(OpT op, bool debugDump) {
  ModelContext model;
  model.logpdfFn = op.getLogpdfFnAttr();
  if (!model.logpdfFn) {
    model.fn = op.getFnAttr();
    model.originalTrace = op.getOriginalTrace();
  }
  model.fnInputs = llvm::to_vector(op.getInputs().drop_front());
  model.allAddresses = op.getAllAddressesAttr();
  model.autodiffAttrs = op.getAutodiffAttrsAttr();
  model.debugDump = debugDump;
  if constexpr (std::is_same_v<OpT, impulse::InferRegionOp>)
    model.logpdfRegion = &op.getLogpdf();
  return model;
}

/// Lays out the top-level sites in the `selection` of `op` in
/// `model.supports`, in `all_addresses` order if `positionDeclarationOrder`,
/// else in `selection` order, and reads the result types of the model `fn` of
/// `op`. Returns the number of values the selected sites hold; fails when
/// there is none or a site cannot be laid out.
template <typename OpT>
static FailureOr<int64_t>
resolveFlatModel(OpT op, bool positionDeclarationOrder,
                 SymbolTableCollection &symbolTable, ModelContext &model) {
  auto fn = cast<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
  ArrayAttr selection = op.getSelectionAttr();
  FailureOr<FlatTraceLayout> selected =
      getFlatTraceLayout(op, fn, selection, symbolTable);
  if (failed(selected) || selected->positionSize <= 0)
    return failure();
  if (failed(verifySelectedSupports(op, *selected)))
    return failure();
  model.supports = collectSupportInfoForSelection(
      op, fn, selection, *selected, model.allAddresses, symbolTable,
      positionDeclarationOrder);
  auto fnType = cast<FunctionType>(fn.getFunctionType());
  model.fnResultTypes.assign(fnType.getResults().begin(),
                             fnType.getResults().end());
  return selected->positionSize;
}

template <typename OpT>
MCMCChainEmitter::MCMCChainEmitter(OpBuilder &builder, OpT op,
                                   ModelContext model)
    : builder(builder), loc(op.getLoc()), model(std::move(model)),
      numWarmup(op.getNumWarmup()), totalWarmup(op.getNumWarmup()),
      numSamples(op.getNumSamples()), thinning(op.getThinning()),
      numChains(op.getNumChains()), schema(op.getStateSchemaAttr()),
      stateIn(op.getStateIn()), initPosition(op.getInitPosition()),
      initStepSize(op.getInitStepSize()),
      initInvMass(op.getInitInverseMassMatrix()),
      isHMC(op.getHmcConfig().has_value()),
      isNUTS(op.getNutsConfig().has_value()) {
  if (auto total = op.getTotalWarmup())
    totalWarmup = *total;

  exposeAdaptation =
      impulse::slotIndex(schema, state_slot::kWindowIdx).has_value();
  if (std::optional<unsigned> first =
          impulse::slotIndex(schema, state_slot::kDaLogStepSize);
      first && *first < stateIn.size()) {
    SmallVector<Value> slots = llvm::to_vector(stateIn.drop_front(*first));
    ArrayRef<Value> values = slots;
    savedAdaptation = AdaptationState::consume(values, /*hasWelford=*/true);
  }

  // Multi-chain slots are chain-major; emitChain reads them per chain.
  if (numChains <= 1) {
    if (Value savedStepSize = stateInSlot(state_slot::kStepSize))
      initStepSize = savedStepSize;
    if (Value savedInvMass = stateInSlot(state_slot::kInverseMassMatrix))
      initInvMass = savedInvMass;
  }

  elemType = cast<RankedTensorType>(initStepSize.getType()).getElementType();
  this->model.elementType = elemType;
  positionType = RankedTensorType::get({1, this->model.positionSize}, elemType);
  scalarType = RankedTensorType::get({}, elemType);
  i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  auto f64TensorType = RankedTensorType::get({}, builder.getF64Type());
  if (isHMC) {
    auto hmcConfig = op.getHmcConfig().value();
    numSteps = hmcConfig.getNumSteps();
    trajectoryLength =
        emitSplatConstant(builder, loc, f64TensorType,
                          hmcConfig.getTrajectoryLength().getValueAsDouble());
    adaptStepSize = hmcConfig.getAdaptStepSize();
    adaptMassMatrix = hmcConfig.getAdaptMassMatrix();
  } else {
    auto nutsConfig = op.getNutsConfig().value();
    maxTreeDepth = nutsConfig.getMaxTreeDepth();
    adaptStepSize = nutsConfig.getAdaptStepSize();
    adaptMassMatrix = nutsConfig.getAdaptMassMatrix();
    double maxDeltaEnergyVal =
        nutsConfig.getMaxDeltaEnergy()
            ? nutsConfig.getMaxDeltaEnergy().getValueAsDouble()
            : 1000.0;
    maxDeltaEnergy =
        emitSplatConstant(builder, loc, f64TensorType, maxDeltaEnergyVal);
  }

  if (initInvMass)
    diagonal = impulse::isDiagonalInverseMass(
        cast<RankedTensorType>(initInvMass.getType()));
  welfordConfig.diagonal = diagonal;
}

SmallVector<ChainResult> MCMCChainEmitter::emitChains(Value rng) {
  if (numChains <= 1)
    return {emitChain(rng, 0)};

  Value savedKeys = stateInSlot(state_slot::kRng);
  impulse::RandomSplitOp splitOp;
  if (!savedKeys) {
    SmallVector<Type> chainKeyTypes(numChains, rng.getType());
    splitOp = impulse::RandomSplitOp::create(builder, loc, chainKeyTypes, rng);
  }

  SmallVector<ChainResult> chains;
  chains.reserve(numChains);
  for (int64_t chainIdx = 0; chainIdx < numChains; ++chainIdx) {
    Value chainRng = savedKeys
                         ? emitLeadingEntry(builder, loc, savedKeys, chainIdx)
                         : splitOp.getResult(chainIdx);
    chains.push_back(emitChain(chainRng, chainIdx));
  }
  return chains;
}

ChainResult MCMCChainEmitter::emitChain(Value rng, int64_t chainIdx) {
  Value stepSize = initStepSize;
  Value invMass = initInvMass;
  std::optional<AdaptationState> saved = savedAdaptation;
  if (numChains > 1) {
    if (Value savedStepSize = stateInSlot(state_slot::kStepSize))
      stepSize = emitLeadingEntry(builder, loc, savedStepSize, chainIdx);
    if (Value savedInvMass = stateInSlot(state_slot::kInverseMassMatrix))
      invMass = emitLeadingEntry(builder, loc, savedInvMass, chainIdx);
    if (saved) {
      SmallVector<Value> slots = saved->toValues();
      for (Value &slot : slots)
        slot = emitLeadingEntry(builder, loc, slot, chainIdx);
      ArrayRef<Value> values = slots;
      saved = AdaptationState::consume(values, /*hasWelford=*/true);
    }
  }

  Value massMatrixSqrt =
      computeMassMatrixSqrt(builder, loc, invMass, positionType);
  ChainPoint state =
      initChainState(rng, chainIdx, stepSize, invMass, massMatrixSqrt);

  if (!invMass) {
    invMass = emitSplatConstant(builder, loc, positionType, 1.0);
    massMatrixSqrt = emitSplatConstant(builder, loc, positionType, 1.0);
  }

  std::optional<AdaptationState> finalAdaptation;
  if (numWarmup > 0) {
    WarmupCarry warmedUp =
        emitWarmup(state, stepSize, invMass, massMatrixSqrt, saved, chainIdx);
    state = warmedUp.chain;
    invMass = warmedUp.invMass;
    massMatrixSqrt = warmedUp.massMatrixSqrt;
    stepSize = conditionalDump(builder, loc, warmedUp.stepSize,
                               "MCMC: adapted step size after warmup",
                               model.debugDump);
    if (adaptMassMatrix) {
      invMass = conditionalDump(
          builder, loc, invMass,
          "MCMC: adapted inverse mass matrix after warmup", model.debugDump);
    }
    if (exposeAdaptation) {
      finalAdaptation = warmedUp.adaptation;
      if (!adaptMassMatrix)
        finalAdaptation->welford =
            initWelford(builder, loc, model.positionSize, diagonal);
    }
  } else if (exposeAdaptation) {
    finalAdaptation =
        loadOrInitAdaptation(stepSize, /*withWelford=*/true, saved);
  }

  ChainResult result = emitSampling(state, stepSize, invMass, massMatrixSqrt);
  result.trace = conditionalDump(builder, loc, result.trace,
                                 "MCMC: collected samples", model.debugDump);
  if (schema && !schema.empty())
    result.stateOut =
        packStateOut(state, stepSize, invMass, finalAdaptation, chainIdx);
  return result;
}

Value MCMCChainEmitter::stateInSlot(StringRef name) const {
  return impulse::stateInSlot(schema, stateIn, name);
}

Value MCMCChainEmitter::chainStateInSlot(StringRef name, int64_t chainIdx) {
  Value value = stateInSlot(name);
  if (value && numChains > 1)
    value = emitLeadingEntry(builder, loc, value, chainIdx);
  return value;
}

HMCContext MCMCChainEmitter::makeHMCContext(Value invMass, Value massMatrixSqrt,
                                            Value stepSize) const {
  return HMCContext{model,          stepSize,         invMass,
                    massMatrixSqrt, trajectoryLength, numSteps};
}

NUTSContext MCMCChainEmitter::makeNUTSContext(Value invMass,
                                              Value massMatrixSqrt,
                                              Value stepSize,
                                              Value potentialEnergy) const {
  return NUTSContext{makeHMCContext(invMass, massMatrixSqrt, stepSize),
                     potentialEnergy, maxDeltaEnergy, maxTreeDepth};
}

MCMCKernelResult MCMCChainEmitter::runKernel(const ChainPoint &state,
                                             Value stepSize, Value invMass,
                                             Value massMatrixSqrt) {
  if (isHMC)
    return SampleHMC(builder, loc, state,
                     makeHMCContext(invMass, massMatrixSqrt, stepSize));
  return SampleNUTS(builder, loc, state,
                    makeNUTSContext(invMass, massMatrixSqrt, stepSize,
                                    state.potentialEnergy));
}

ChainPoint MCMCChainEmitter::initChainState(Value rng, int64_t chainIdx,
                                            Value stepSize, Value invMass,
                                            Value massMatrixSqrt) {
  Value position = chainStateInSlot(state_slot::kPosition, chainIdx);
  Value grad = chainStateInSlot(state_slot::kGradient, chainIdx);
  Value potentialEnergy =
      chainStateInSlot(state_slot::kPotentialEnergy, chainIdx);
  if (position && grad && potentialEnergy)
    return {position, grad, potentialEnergy, rng};

  HMCContext ctx = makeHMCContext(invMass, massMatrixSqrt, stepSize);
  Value chainInitPosition;
  if (model.logpdfFn) {
    chainInitPosition = initPosition;
    if (numChains > 1)
      chainInitPosition =
          emitLeadingEntry(builder, loc, chainInitPosition, chainIdx);
  }
  return InitHMC(builder, loc, rng, ctx, chainInitPosition);
}

AdaptationState MCMCChainEmitter::loadOrInitAdaptation(
    Value stepSize, bool withWelford,
    const std::optional<AdaptationState> &saved) {
  if (saved)
    return *saved;
  AdaptationState fresh;
  fresh.dualAveraging =
      initDualAveraging(builder, loc, stepSize, /*logOfProduct=*/true);
  if (withWelford)
    fresh.welford = initWelford(builder, loc, model.positionSize, diagonal);
  fresh.windowIdx = emitI64ScalarConstant(builder, loc, 0);
  return fresh;
}

WarmupCarry
MCMCChainEmitter::emitWarmup(const ChainPoint &start, Value stepSize,
                             Value invMass, Value massMatrixSqrt,
                             const std::optional<AdaptationState> &saved,
                             int64_t chainIdx) {
  WarmupSchedule schedule;
  schedule.zero = emitI64ScalarConstant(builder, loc, 0);
  schedule.one = emitI64ScalarConstant(builder, loc, 1);
  Value numWarmupConst = emitI64ScalarConstant(builder, loc, numWarmup);

  SmallVector<AdaptWindow> windows = buildAdaptationSchedule(totalWarmup);
  int64_t numWindows = static_cast<int64_t>(windows.size());
  for (const auto &window : windows)
    schedule.windowEnds.push_back(
        emitI64ScalarConstant(builder, loc, window.end));
  schedule.lastWindow = emitI64ScalarConstant(builder, loc, numWindows - 1);
  schedule.lastIteration = emitI64ScalarConstant(builder, loc, totalWarmup - 1);

  Value initialStepSize =
      conditionalDump(builder, loc, stepSize,
                      "MCMC: initial step size before warmup", model.debugDump);
  AdaptationState adaptation =
      loadOrInitAdaptation(initialStepSize, adaptMassMatrix, saved);
  if (!adaptMassMatrix)
    adaptation.welford.reset();

  WarmupCarry init;
  init.chain = start;
  init.stepSize = initialStepSize;
  init.invMass = invMass;
  init.massMatrixSqrt = massMatrixSqrt;
  init.adaptation = adaptation;
  SmallVector<Type> warmupLoopTypes = init.getTypes(positionType, scalarType);

  auto warmupLoop =
      impulse::ForOp::create(builder, loc, warmupLoopTypes, schedule.zero,
                             numWarmupConst, schedule.one, init.toValues());

  Block *warmupBody = createForBody(builder, warmupLoop);
  Value iteration = warmupBody->getArgument(0);
  if (Value warmupOffset =
          chainStateInSlot(state_slot::kWarmupOffset, chainIdx))
    iteration = arith::AddIOp::create(builder, loc, warmupOffset, iteration);
  WarmupCarry carry = WarmupCarry::fromValues(
      warmupBody->getArguments().drop_front(), adaptMassMatrix);
  WarmupCarry next = emitWarmupStep(carry, iteration, schedule);
  impulse::YieldOp::create(builder, loc, next.toValues());

  builder.setInsertionPointAfter(warmupLoop);
  return WarmupCarry::fromValues(warmupLoop.getResults(), adaptMassMatrix);
}

WarmupCarry MCMCChainEmitter::emitWarmupStep(const WarmupCarry &carry,
                                             Value iteration,
                                             const WarmupSchedule &schedule) {
  auto sample = runKernel(carry.chain, carry.stepSize, carry.invMass,
                          carry.massMatrixSqrt);

  DualAveragingConfig daConfig;
  DualAveragingState updatedDaState;
  Value currentStepSizeFromDA;
  Value finalStepSizeFromDA;

  if (adaptStepSize) {
    updatedDaState =
        updateDualAveraging(builder, loc, carry.adaptation.dualAveraging,
                            sample.acceptProb, daConfig);
    currentStepSizeFromDA =
        getStepSizeFromDualAveraging(builder, loc, updatedDaState);
    finalStepSizeFromDA =
        getStepSizeFromDualAveraging(builder, loc, updatedDaState, true);
  } else {
    updatedDaState = carry.adaptation.dualAveraging;
    currentStepSizeFromDA = carry.stepSize;
    finalStepSizeFromDA = carry.stepSize;
  }

  auto isLastIter =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, iteration,
                            schedule.lastIteration);
  Value adaptedStepSizeInLoop =
      impulse::SelectOp::create(builder, loc, scalarType, isLastIter,
                                finalStepSizeFromDA, currentStepSizeFromDA);

  const auto &floatSemantics = cast<FloatType>(elemType).getFloatSemantics();
  auto tinyConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(
          scalarType, FloatAttr::get(elemType, llvm::APFloat::getSmallest(
                                                   floatSemantics))));
  auto maxConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(
          scalarType,
          FloatAttr::get(elemType, llvm::APFloat::getLargest(floatSemantics))));
  adaptedStepSizeInLoop =
      arith::MaximumFOp::create(builder, loc, adaptedStepSizeInLoop, tinyConst);
  adaptedStepSizeInLoop =
      arith::MinimumFOp::create(builder, loc, adaptedStepSizeInLoop, maxConst);

  auto windowIdxGtZero =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                            carry.adaptation.windowIdx, schedule.zero);
  auto windowIdxLtLast =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                            carry.adaptation.windowIdx, schedule.lastWindow);
  auto isMiddleWindow =
      arith::AndIOp::create(builder, loc, windowIdxGtZero, windowIdxLtLast);

  WelfordState conditionalWelford;
  if (adaptMassMatrix) {
    auto sampleType1D = RankedTensorType::get({model.positionSize}, elemType);
    Value sample1D =
        impulse::ReshapeOp::create(builder, loc, sampleType1D, sample.next.q);
    const WelfordState &welford = *carry.adaptation.welford;
    WelfordState updatedWelfordAfterSample =
        updateWelford(builder, loc, welford, sample1D, welfordConfig);

    conditionalWelford.mean = impulse::SelectOp::create(
        builder, loc, welford.mean.getType(), isMiddleWindow,
        updatedWelfordAfterSample.mean, welford.mean);
    conditionalWelford.m2 = impulse::SelectOp::create(
        builder, loc, welford.m2.getType(), isMiddleWindow,
        updatedWelfordAfterSample.m2, welford.m2);
    conditionalWelford.n = impulse::SelectOp::create(
        builder, loc, welford.n.getType(), isMiddleWindow,
        updatedWelfordAfterSample.n, welford.n);
  }

  Value atWindowEnd = emitI1ScalarConstant(builder, loc, false);

  for (auto [window, windowEnd] : llvm::enumerate(schedule.windowEnds)) {
    auto inWindow = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, carry.adaptation.windowIdx,
        emitI64ScalarConstant(builder, loc, window));
    auto atEnd = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       iteration, windowEnd);
    auto matchesThisWindow =
        arith::AndIOp::create(builder, loc, inWindow, atEnd);
    atWindowEnd =
        arith::OrIOp::create(builder, loc, atWindowEnd, matchesThisWindow);
  }

  Value newWindowIdx = arith::AddIOp::create(
      builder, loc, carry.adaptation.windowIdx, schedule.one);
  Value windowIdxAfterIncrement =
      impulse::SelectOp::create(builder, loc, i64TensorType, atWindowEnd,
                                newWindowIdx, carry.adaptation.windowIdx);

  auto atMiddleWindowEnd =
      arith::AndIOp::create(builder, loc, atWindowEnd, isMiddleWindow);

  WarmupCarry next;
  next.chain = sample.next;
  next.stepSize = adaptedStepSizeInLoop;
  next.invMass = carry.invMass;
  next.massMatrixSqrt = carry.massMatrixSqrt;
  next.adaptation.dualAveraging = updatedDaState;
  if (adaptMassMatrix)
    next.adaptation.welford = conditionalWelford;
  next.adaptation.windowIdx = windowIdxAfterIncrement;

  emitWindowEndRestart(next, atMiddleWindowEnd);
  return next;
}

void MCMCChainEmitter::emitWindowEndRestart(WarmupCarry &next,
                                            Value atMiddleWindowEnd) {
  auto windowEndValues = [](const WarmupCarry &c) {
    SmallVector<Value> values = {c.invMass, c.massMatrixSqrt};
    if (c.adaptation.welford)
      llvm::append_range(values, c.adaptation.welford->toValues());
    llvm::append_range(values, c.adaptation.dualAveraging.toValues());
    return values;
  };
  SmallVector<Value> keptValues = windowEndValues(next);
  auto ifOp = impulse::IfOp::create(
      builder, loc, ValueRange(keptValues).getTypes(), atMiddleWindowEnd);

  builder.setInsertionPointToStart(builder.createBlock(&ifOp.getTrueBranch()));
  WarmupCarry restarted = next;
  if (adaptMassMatrix) {
    Value newInvMass =
        finalizeWelford(builder, loc, *next.adaptation.welford, welfordConfig);
    Value newMassMatrixSqrt =
        computeMassMatrixSqrt(builder, loc, newInvMass, positionType);
    // finalizeWelford returns rank-1 diagonals; match the carried type.
    if (newInvMass.getType() != next.invMass.getType())
      newInvMass = impulse::ReshapeOp::create(
          builder, loc, next.invMass.getType(), newInvMass);
    if (newMassMatrixSqrt.getType() != next.massMatrixSqrt.getType())
      newMassMatrixSqrt = impulse::ReshapeOp::create(
          builder, loc, next.massMatrixSqrt.getType(), newMassMatrixSqrt);
    restarted.invMass = newInvMass;
    restarted.massMatrixSqrt = newMassMatrixSqrt;
    restarted.adaptation.welford =
        initWelford(builder, loc, model.positionSize, diagonal);
  }
  if (adaptStepSize)
    restarted.adaptation.dualAveraging =
        initDualAveraging(builder, loc, next.stepSize);
  impulse::YieldOp::create(builder, loc, windowEndValues(restarted));

  builder.setInsertionPointToStart(builder.createBlock(&ifOp.getFalseBranch()));
  impulse::YieldOp::create(builder, loc, keptValues);

  builder.setInsertionPointAfter(ifOp);
  SmallVector<Value> results = llvm::to_vector_of<Value>(ifOp.getResults());
  ArrayRef<Value> updated = results;
  next.invMass = updated[0];
  next.massMatrixSqrt = updated[1];
  updated = updated.drop_front(2);
  if (adaptMassMatrix) {
    next.adaptation.welford =
        WelfordState::fromValues(updated.take_front(WelfordState::kNumValues));
    updated = updated.drop_front(WelfordState::kNumValues);
  }
  next.adaptation.dualAveraging = DualAveragingState::fromValues(updated);
}

ChainResult MCMCChainEmitter::emitSampling(ChainPoint &state, Value stepSize,
                                           Value invMass,
                                           Value massMatrixSqrt) {
  int64_t collectionSize = numSamples / thinning;
  int64_t startIdx = numSamples % thinning;

  auto samplesBufferType = RankedTensorType::get(
      {collectionSize, model.getConstrainedPositionSize()}, elemType);
  auto diagnosticsBufferType =
      RankedTensorType::get({collectionSize, 2}, builder.getI1Type());
  auto logDensityBufferType = RankedTensorType::get({collectionSize}, elemType);

  Value samplesBuffer = emitSplatConstant(builder, loc, samplesBufferType, 0.0);
  auto diagnosticsBuffer = arith::ConstantOp::create(
      builder, loc, diagnosticsBufferType,
      DenseElementsAttr::get(diagnosticsBufferType,
                             builder.getBoolAttr(isNUTS)));
  Value logDensityBuffer =
      emitSplatConstant(builder, loc, logDensityBufferType, 0.0);

  Value zero = emitI64ScalarConstant(builder, loc, 0);
  Value one = emitI64ScalarConstant(builder, loc, 1);
  Value numSamplesConst = emitI64ScalarConstant(builder, loc, numSamples);
  Value startIdxConst = emitI64ScalarConstant(builder, loc, startIdx);
  Value thinningConst = emitI64ScalarConstant(builder, loc, thinning);

  if (collectionSize == 0)
    return {samplesBuffer, diagnosticsBuffer, logDensityBuffer, state.rng, {}};

  SmallVector<Type> loopResultTypes = {
      positionType,        positionType,      scalarType,
      state.rng.getType(), samplesBufferType, diagnosticsBufferType,
      logDensityBufferType};
  auto forLoopOp = impulse::ForOp::create(
      builder, loc, loopResultTypes, zero, numSamplesConst, one,
      ValueRange{state.q, state.grad, state.potentialEnergy, state.rng,
                 samplesBuffer, diagnosticsBuffer, logDensityBuffer});

  Block *loopBody = createForBody(builder, forLoopOp);
  Value iterIdx = loopBody->getArgument(0);
  ChainPoint chainLoop = {loopBody->getArgument(1), loopBody->getArgument(2),
                          loopBody->getArgument(3), loopBody->getArgument(4)};
  Value samplesBufferLoop = loopBody->getArgument(5);
  Value diagnosticsBufferLoop = loopBody->getArgument(6);
  Value logDensityBufferLoop = loopBody->getArgument(7);

  auto sample = runKernel(chainLoop, stepSize, invMass, massMatrixSqrt);
  auto qConstrained =
      impulse::constrainPosition(builder, loc, sample.next.q, model.supports);

  auto iMinusStart =
      arith::SubIOp::create(builder, loc, iterIdx, startIdxConst);
  auto storageIdx =
      arith::DivSIOp::create(builder, loc, iMinusStart, thinningConst);

  auto geStartIdx = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sge, iterIdx, startIdxConst);
  auto modThinning =
      arith::RemSIOp::create(builder, loc, iMinusStart, thinningConst);
  auto modIsZero = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         modThinning, zero);
  auto shouldStore = arith::AndIOp::create(builder, loc, geStartIdx, modIsZero);

  Value zeroCol = emitI64ScalarConstant(builder, loc, 0);
  auto updatedSamplesBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, samplesBufferType, samplesBufferLoop, qConstrained,
      ValueRange{storageIdx, zeroCol});
  auto selectedSamplesBuffer =
      impulse::SelectOp::create(builder, loc, samplesBufferType, shouldStore,
                                updatedSamplesBuffer, samplesBufferLoop);

  Value oneCol = emitI64ScalarConstant(builder, loc, 1);
  auto accepted1x1 = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({1, 1}, builder.getI1Type()),
      sample.accepted);
  auto divergent1x1 = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({1, 1}, builder.getI1Type()),
      sample.divergent);
  auto updatedDiagnosticsBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, diagnosticsBufferType, diagnosticsBufferLoop, accepted1x1,
      ValueRange{storageIdx, zeroCol});
  updatedDiagnosticsBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, diagnosticsBufferType, updatedDiagnosticsBuffer,
      divergent1x1, ValueRange{storageIdx, oneCol});
  auto selectedDiagnosticsBuffer = impulse::SelectOp::create(
      builder, loc, diagnosticsBufferType, shouldStore,
      updatedDiagnosticsBuffer, diagnosticsBufferLoop);

  auto negU = arith::NegFOp::create(builder, loc, sample.next.potentialEnergy);
  auto logDensity1D = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({1}, elemType), negU);
  auto updatedLogDensityBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, logDensityBufferType, logDensityBufferLoop, logDensity1D,
      ValueRange{storageIdx});
  auto selectedLogDensityBuffer =
      impulse::SelectOp::create(builder, loc, logDensityBufferType, shouldStore,
                                updatedLogDensityBuffer, logDensityBufferLoop);

  impulse::YieldOp::create(
      builder, loc,
      ValueRange{sample.next.q, sample.next.grad, sample.next.potentialEnergy,
                 sample.next.rng, selectedSamplesBuffer,
                 selectedDiagnosticsBuffer, selectedLogDensityBuffer});

  builder.setInsertionPointAfter(forLoopOp);
  state = {forLoopOp.getResult(0), forLoopOp.getResult(1),
           forLoopOp.getResult(2), forLoopOp.getResult(3)};
  return {forLoopOp.getResult(4),
          forLoopOp.getResult(5),
          forLoopOp.getResult(6),
          state.rng,
          {}};
}

SmallVector<Value> MCMCChainEmitter::packStateOut(
    const ChainPoint &state, Value stepSize, Value invMass,
    const std::optional<AdaptationState> &adaptation, int64_t chainIdx) {
  Value warmupOffsetIn = chainStateInSlot(state_slot::kWarmupOffset, chainIdx);
  if (!warmupOffsetIn)
    warmupOffsetIn = emitI64ScalarConstant(builder, loc, 0);
  Value warmupOffsetOut =
      arith::AddIOp::create(builder, loc, warmupOffsetIn,
                            emitI64ScalarConstant(builder, loc, numWarmup));

  SmallVector<Value> stateOut = {state.q, state.grad, state.potentialEnergy,
                                 stepSize, invMass};
  if (adaptation) {
    llvm::append_range(stateOut, adaptation->toValues());
    stateOut.push_back(warmupOffsetOut);
  }
  return stateOut;
}

namespace {

/// Lowers an `impulse.infer` or `impulse.infer_region` with an HMC or NUTS
/// config to its chains.
template <typename MCMCOpT>
struct LowerMCMCPattern : public mlir::OpRewritePattern<MCMCOpT> {
  static constexpr bool isRegionForm =
      std::is_same_v<MCMCOpT, impulse::InferRegionOp>;

  bool debugDump;

  LowerMCMCPattern(MLIRContext *context, bool debugDump,
                   PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<MCMCOpT>(context, benefit),
        debugDump(debugDump) {}

  /// Reads the model of an HMC/NUTS op. Reports a match failure for other
  /// inference configs and emits an error for a malformed op.
  FailureOr<ModelContext> resolveModel(MCMCOpT mcmcOp,
                                       PatternRewriter &rewriter) const {
    SymbolTableCollection symbolTable;

    bool hasLogpdfFn = static_cast<bool>(mcmcOp.getLogpdfFnAttr());

    if constexpr (isRegionForm) {
      if (mcmcOp.getLogpdf().empty()) {
        mcmcOp.emitError("Impulse: infer_region requires a constructed "
                         "logpdf region (run sicm first)");
        return failure();
      }
      if (!hasLogpdfFn && mcmcOp.getSampling().empty()) {
        mcmcOp.emitError("Impulse: model-mode infer_region requires a "
                         "sampler region");
        return failure();
      }
    } else if (!hasLogpdfFn) {
      auto fnAttr = mcmcOp.getFnAttr();
      if (!fnAttr) {
        mcmcOp.emitError("Impulse: either fn or logpdf_fn must be provided");
        return failure();
      }
      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp, fnAttr));
      if (fn.getFunctionBody().empty()) {
        mcmcOp.emitError("Impulse: calling `infer` on an empty function");
        return failure();
      }
    }

    // Other configs fall through to their own patterns; do not error here.
    if (!mcmcOp.getHmcConfig() && !mcmcOp.getNutsConfig())
      return rewriter.notifyMatchFailure(
          mcmcOp, "no hmc_config/nuts_config; not an HMC/NUTS op");

    if (!mcmcOp.getInitStepSize()) {
      mcmcOp.emitError("Impulse: MCMC requires init_step_size parameter");
      return failure();
    }

    if (mcmcOp.getInputs().empty()) {
      mcmcOp.emitError("Impulse: MCMC requires at least rng_state input");
      return failure();
    }

    ModelContext model = readInferModel(mcmcOp, debugDump);
    if constexpr (isRegionForm) {
      // The logpdf region takes the position values in their order.
      int64_t column = 0;
      for (BlockArgument arg : mcmcOp.getLogpdf().front().getArguments()) {
        model.logpdfArgOffsets.push_back(column);
        column += cast<RankedTensorType>(arg.getType()).getNumElements();
      }
    }
    auto unconstrainedSizeOf = [](ArrayRef<impulse::SupportInfo> supports) {
      int64_t size = 0;
      for (const impulse::SupportInfo &info : supports)
        size += info.unconstrainedSize;
      return size;
    };

    if (hasLogpdfFn) {
      auto initPosType =
          cast<RankedTensorType>(mcmcOp.getInitPosition().getType());
      // `init_position` is `1xN` or chain-major `Cx1xN`; N is per chain.
      model.positionSize = initPosType.getNumElements() / mcmcOp.getNumChains();
    } else if constexpr (isRegionForm) {
      if (failed(verifySelectedSupports(mcmcOp)))
        return failure();
      FailureOr<RegionPositionLayout> layout =
          collectRegionLayout(mcmcOp, /*positionDeclarationOrder=*/false);
      if (succeeded(layout))
        model.supports = std::move(layout->supports);
      if (model.supports.empty()) {
        mcmcOp.emitError("Impulse: failed to collect per-site support info "
                         "from the infer_region sampler");
        return failure();
      }
      model.positionSize = unconstrainedSizeOf(model.supports);
      if (model.positionSize <= 0)
        return failure();
    } else {
      for (Attribute address : mcmcOp.getSelectionAttr()) {
        if (cast<ArrayAttr>(address).size() > 1) {
          mcmcOp.emitError("Impulse: HMC and NUTS on a model take only "
                           "top-level addresses in `selection`, not ")
              << address;
          return failure();
        }
      }
      if (failed(resolveFlatModel(mcmcOp, /*positionDeclarationOrder=*/false,
                                  symbolTable, model)))
        return failure();
      model.positionSize = unconstrainedSizeOf(model.supports);
    }
    return model;
  }

  LogicalResult matchAndRewrite(MCMCOpT mcmcOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<ModelContext> model = resolveModel(mcmcOp, rewriter);
    if (failed(model))
      return failure();

    MCMCChainEmitter emitter(rewriter, mcmcOp, std::move(*model));
    SmallVector<ChainResult> chains =
        emitter.emitChains(mcmcOp.getInputs().front());
    if (chains.size() == 1) {
      rewriter.replaceOp(mcmcOp, chains.front().toValues());
      return success();
    }
    ArrayAttr schema = mcmcOp.getStateSchemaAttr();
    rewriter.replaceOp(mcmcOp,
                       stackChainResults(rewriter, mcmcOp.getLoc(), chains,
                                         schema && !schema.empty()));
    return success();
  }
};

/// Lowers an `impulse.infer` or `impulse.infer_region` with a VI config to
/// an SVI loop whose learning rate is the `init_step_size` operand.
template <typename OpT>
struct LowerVIPattern : public mlir::OpRewritePattern<OpT> {
  static constexpr bool isRegionForm =
      std::is_same_v<OpT, impulse::InferRegionOp>;

  bool debugDump;

  LowerVIPattern(MLIRContext *context, bool debugDump,
                 PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<OpT>(context, benefit), debugDump(debugDump) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    if (!op.getViConfig().has_value())
      return rewriter.notifyMatchFailure(op, "no vi_config; not a VI op");

    SymbolTableCollection symbolTable;
    bool hasLogpdfFn = static_cast<bool>(op.getLogpdfFnAttr());

    if (!op.getInitStepSize()) {
      op.emitError("Impulse VI: requires init_step_size (learning rate)");
      return failure();
    }

    if (op.getInputs().empty()) {
      op.emitError("Impulse VI: requires at least rng_state input");
      return failure();
    }

    ModelContext model = readInferModel(op, debugDump);
    if constexpr (isRegionForm) {
      if (hasLogpdfFn) {
        op.emitError("Impulse VI: custom-logpdf mode is not supported in "
                     "region form");
        return failure();
      }
      if (op.getLogpdf().empty()) {
        op.emitError("Impulse VI: infer_region requires a constructed logpdf "
                     "region (run sicm first)");
        return failure();
      }
      if (failed(verifySelectedSupports(op)))
        return failure();
      // Per-site draws consume the RNG in program order, so the position
      // follows `all_addresses`.
      FailureOr<RegionPositionLayout> layout =
          collectRegionLayout(op, /*positionDeclarationOrder=*/true);
      if (failed(layout)) {
        op.emitError("Impulse VI: failed to derive position layout from the "
                     "infer_region sampler");
        return failure();
      }
      model.supports = std::move(layout->supports);
      model.logpdfArgOffsets = std::move(layout->argOffsets);
      model.positionSize = model.getConstrainedPositionSize();
      if (model.positionSize <= 0)
        return failure();
    } else if (hasLogpdfFn) {
      model.positionSize =
          cast<RankedTensorType>(op.getInitPosition().getType())
              .getNumElements();
    } else {
      FailureOr<int64_t> numValues = resolveFlatModel(
          op, /*positionDeclarationOrder=*/true, symbolTable, model);
      if (failed(numValues))
        return failure();
      model.positionSize = *numValues;
    }

    return lowerVI(rewriter, op, std::move(model));
  }
};

} // namespace

void impulse::populateExpandImpulseHMCVIPatterns(RewritePatternSet &patterns,
                                                 bool debugDump) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerMCMCPattern<impulse::InferOp>,
               LowerMCMCPattern<impulse::InferRegionOp>>(context, debugDump);
  patterns.add<LowerVIPattern<impulse::InferOp>,
               LowerVIPattern<impulse::InferRegionOp>>(context, debugDump);
}
