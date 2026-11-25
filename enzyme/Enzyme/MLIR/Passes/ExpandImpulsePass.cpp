//===- ExpandImpulsePass.cpp - Lower the Impulse inference ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the expand-impulse pass, which lowers the Impulse
// inference ops (simulate, generate, regenerate, untraced_call, mh, and infer
// and infer_region with an HMC, NUTS, VI, MH or Gibbs configuration) into
// calls, arith and math ops and lower-level Impulse ops.
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Impulse/InferStateLayout.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "Interfaces/HMCUtils.h"
#include "Interfaces/ImpulseUtils.h"
#include "Interfaces/TransformUtils.h"
#include "Interfaces/VIUtils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/BitVector.h"

#include <type_traits>

#define DEBUG_TYPE "expand-impulse"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_EXPANDIMPULSEPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

static int64_t computeTensorElementCount(RankedTensorType tensorType) {
  int64_t elemCount = 1;
  for (auto dim : tensorType.getShape()) {
    if (dim == ShapedType::kDynamic)
      return -1;
    elemCount *= dim;
  }
  return elemCount;
}

using SampleOpMap = DenseMap<Attribute, impulse::SampleOp>;

static SampleOpMap buildSampleOpMap(FunctionOpInterface fn) {
  SampleOpMap map;
  fn.walk([&](impulse::SampleOp sampleOp) {
    if (auto symbol = sampleOp.getSymbolAttr())
      map[symbol] = sampleOp;
  });
  return map;
}

static impulse::SampleOp findSampleBySymbol(const SampleOpMap &map,
                                            Attribute targetSymbol) {
  auto it = map.find(targetSymbol);
  return it != map.end() ? it->second : nullptr;
}

static int64_t computeSampleElementCount(Operation *op,
                                         impulse::SampleOp sampleOp) {
  int64_t totalCount = 0;
  for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
    auto resultType = sampleOp.getResult(i).getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType) {
      op->emitError("Expected ranked tensor type for sample result");
      return -1;
    }
    int64_t elemCount = computeTensorElementCount(tensorType);
    if (elemCount < 0) {
      op->emitError("Dynamic tensor dimensions not supported");
      return -1;
    }
    totalCount += elemCount;
  }
  return totalCount;
}

static bool computePositionSizeForAddress(Operation *op,
                                          const SampleOpMap &sampleMap,
                                          ArrayRef<Attribute> address,
                                          SymbolTableCollection &symbolTable,
                                          int64_t &positionSize) {
  if (address.empty())
    return false;

  auto sampleOp = findSampleBySymbol(sampleMap, address[0]);
  if (!sampleOp)
    return false;

  if (address.size() > 1) {
    if (sampleOp.getLogpdfFnAttr()) {
      op->emitError("Cannot select nested address in distribution function");
      return false;
    }

    auto genFn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
    if (!genFn || genFn.getFunctionBody().empty()) {
      op->emitError("Cannot find generative function for nested address");
      return false;
    }

    auto nestedMap = buildSampleOpMap(genFn);
    return computePositionSizeForAddress(op, nestedMap, address.drop_front(),
                                         symbolTable, positionSize);
  }

  int64_t elemCount = computeSampleElementCount(op, sampleOp);
  if (elemCount < 0)
    return false;

  positionSize += elemCount;
  return true;
}

static int64_t
computePositionSizeForSelection(Operation *op, FunctionOpInterface fn,
                                ArrayAttr selection,
                                SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t positionSize = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty()) {
      op->emitError("Empty address in selection");
      return -1;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, sampleMap, tailAddresses,
                                       symbolTable, positionSize)) {
      op->emitError("Could not find sample with symbol in address chain");
      return -1;
    }
  }

  return positionSize;
}

static int64_t
computeOffsetForSampleInSelection(Operation *op, FunctionOpInterface fn,
                                  ArrayAttr selection, Attribute targetSymbol,
                                  SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t offset = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;

    auto firstSymbol = address[0];

    if (firstSymbol == targetSymbol) {
      return offset;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, sampleMap, tailAddresses,
                                       symbolTable, offset)) {
      return -1;
    }
  }

  return -1;
}

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

/// Checks the supports of the sites of `fn` at the top-level addresses in
/// `selection`.
static LogicalResult verifySelectedSupports(Operation *op,
                                            FunctionOpInterface fn,
                                            ArrayAttr selection) {
  SampleOpMap sampleMap = buildSampleOpMap(fn);
  for (Attribute addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.size() != 1)
      continue;
    impulse::SampleOp site = findSampleBySymbol(sampleMap, address[0]);
    if (site && failed(verifyUnconstrainable(op, addr, site.getSupportAttr())))
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

/// Support info for the top-level addresses in `selection`. Positions follow
/// `all_addresses` order when `positionDeclarationOrder` is set, else
/// `selection` order.
static SmallVector<impulse::SupportInfo>
collectSupportInfoForSelection(Operation *op, FunctionOpInterface fn,
                               ArrayAttr selection, ArrayAttr allAddresses,
                               SymbolTableCollection &symbolTable,
                               bool positionDeclarationOrder) {
  auto sampleMap = buildSampleOpMap(fn);
  SmallVector<impulse::SupportInfo> supports;
  int64_t currentPositionOffset = 0;
  int64_t currentUnconstrainedOffset = 0;

  SmallVector<Attribute> orderedSelected;
  if (positionDeclarationOrder) {
    llvm::SmallDenseSet<Attribute, 4> selectedTopLevel;
    for (auto addr : selection) {
      auto address = cast<ArrayAttr>(addr);
      if (address.empty() || address.size() != 1)
        continue;
      selectedTopLevel.insert(address[0]);
    }
    for (auto addr : allAddresses) {
      auto address = cast<ArrayAttr>(addr);
      if (address.empty() || address.size() != 1)
        continue;
      if (!selectedTopLevel.contains(address[0]))
        continue;
      orderedSelected.push_back(address[0]);
    }
  } else {
    for (auto addr : selection) {
      auto address = cast<ArrayAttr>(addr);
      if (address.empty() || address.size() != 1)
        continue;
      orderedSelected.push_back(address[0]);
    }
  }

  for (Attribute targetSymbol : orderedSelected) {
    auto sampleOp = findSampleBySymbol(sampleMap, targetSymbol);
    if (!sampleOp)
      continue;

    auto supportAttr = sampleOp.getSupportAttr();

    int64_t sampleSize = computeSampleElementCount(op, sampleOp);
    if (sampleSize < 0)
      continue;

    int64_t unconstrainedSize =
        transforms::getUnconstrainedSize(sampleSize, supportAttr);

    int64_t traceOffset = computeOffsetForSampleInSelection(
        op, fn, allAddresses, targetSymbol, symbolTable);
    if (traceOffset < 0) {
      op->emitError("Symbol in selection not found in all_addresses - cannot "
                    "determine trace offset for scattered selection");
      return {};
    }

    supports.emplace_back(currentPositionOffset, traceOffset, sampleSize,
                          currentUnconstrainedOffset, unconstrainedSize,
                          supportAttr);
    currentPositionOffset += sampleSize;
    currentUnconstrainedOffset += unconstrainedSize;
  }

  return supports;
}

/// Returns the number of trace columns the values of `site` take.
static int64_t sampleRegionSize(impulse::SampleRegionOp site) {
  int64_t size = 0;
  for (Value result : llvm::drop_begin(site->getResults()))
    size += computeTensorElementCount(cast<RankedTensorType>(result.getType()));
  return size;
}

namespace {
/// Where the sites of a region-form infer op sit on its trace.
struct RegionTraceLayout {
  DenseMap<Attribute, impulse::SampleRegionOp> siteByAddress;
  /// First trace column of each address in `all_addresses`.
  DenseMap<Attribute, int64_t> columnOf;
  /// Columns taken by the values of all addresses, before any score tail.
  int64_t positionSize = 0;
};
} // namespace

/// Lays out the `all_addresses` of `regionOp` on its trace, in that order.
/// Sites are looked up in the whole sampling region, or with `topLevelOnly`
/// in its entry block only. Fails when an address has no site.
static FailureOr<RegionTraceLayout>
getRegionTraceLayout(impulse::InferRegionOp regionOp, bool topLevelOnly) {
  RegionTraceLayout layout;
  auto addSite = [&](impulse::SampleRegionOp site) {
    if (Attribute address = site.getAddressAttr())
      layout.siteByAddress[address] = site;
  };
  if (topLevelOnly)
    for (auto site :
         regionOp.getSampling().front().getOps<impulse::SampleRegionOp>())
      addSite(site);
  else
    regionOp.getSampling().walk(addSite);
  for (Attribute address : regionOp.getAllAddressesAttr()) {
    impulse::SampleRegionOp site = layout.siteByAddress.lookup(address);
    if (!site)
      return failure();
    layout.columnOf[address] = layout.positionSize;
    layout.positionSize += sampleRegionSize(site);
  }
  return layout;
}

/// Position offsets follow `selection`, trace offsets `all_addresses`.
static SmallVector<impulse::SupportInfo>
collectSupportInfoForRegion(impulse::InferRegionOp regionOp) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  if (!selection || !regionOp.getAllAddressesAttr() ||
      regionOp.getSampling().empty())
    return {};
  // Branch-local sites also hold trace columns, so walk the whole sampler.
  FailureOr<RegionTraceLayout> layout =
      getRegionTraceLayout(regionOp, /*topLevelOnly=*/false);
  if (failed(layout))
    return {};

  SmallVector<impulse::SupportInfo> supports;
  int64_t positionOffset = 0;
  int64_t unconstrainedOffset = 0;
  for (Attribute addr : selection) {
    impulse::SampleRegionOp site = layout->siteByAddress.lookup(addr);
    if (!site || !layout->columnOf.count(addr))
      return {};
    int64_t constrainedSize = sampleRegionSize(site);
    int64_t unconstrainedSize = transforms::getUnconstrainedSize(
        constrainedSize, site.getSupportAttr());
    supports.emplace_back(positionOffset, layout->columnOf.lookup(addr),
                          constrainedSize, unconstrainedOffset,
                          unconstrainedSize, site.getSupportAttr());
    positionOffset += constrainedSize;
    unconstrainedOffset += unconstrainedSize;
  }
  return supports;
}

namespace {
/// VI positions in `all_addresses` (declaration) order, so per-site draws
/// consume the RNG in program order; `argSlices` are in `selection` order.
struct RegionVILayout {
  SmallVector<impulse::SupportInfo> supports;
  SmallVector<std::pair<int64_t, int64_t>> argSlices;
  int64_t positionSize = 0;
};
} // namespace

/// Fills `out` with the VI position layout of `regionOp`. Returns false when
/// an address in `selection` or `all_addresses` has no site in the sampler.
static bool collectRegionVILayout(impulse::InferRegionOp regionOp,
                                  RegionVILayout &out) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || !allAddresses || regionOp.getSampling().empty())
    return false;
  FailureOr<RegionTraceLayout> layout =
      getRegionTraceLayout(regionOp, /*topLevelOnly=*/false);
  if (failed(layout))
    return false;

  DenseSet<Attribute> selectedSet(selection.begin(), selection.end());
  DenseMap<Attribute, int64_t> declPositionOffset;
  int64_t positionOffset = 0, unconstrainedOffset = 0;
  for (Attribute addr : allAddresses) {
    if (!selectedSet.contains(addr))
      continue;
    impulse::SampleRegionOp site = layout->siteByAddress.lookup(addr);
    int64_t siteSize = sampleRegionSize(site);
    int64_t unconstrainedSize =
        transforms::getUnconstrainedSize(siteSize, site.getSupportAttr());
    out.supports.emplace_back(positionOffset, layout->columnOf.lookup(addr),
                              siteSize, unconstrainedOffset, unconstrainedSize,
                              site.getSupportAttr());
    declPositionOffset[addr] = positionOffset;
    positionOffset += siteSize;
    unconstrainedOffset += unconstrainedSize;
  }
  out.positionSize = positionOffset;

  for (Attribute addr : selection) {
    impulse::SampleRegionOp site = layout->siteByAddress.lookup(addr);
    if (!site)
      return false;
    int64_t within = 0;
    for (Value result : llvm::drop_begin(site->getResults())) {
      int64_t size = cast<RankedTensorType>(result.getType()).getNumElements();
      out.argSlices.emplace_back(declPositionOffset[addr] + within, size);
      within += size;
    }
  }
  return true;
}

namespace {
/// HMC/NUTS warmup adaptation state, in `state_schema` slot order from
/// `da_log_step_size` through `window_idx`.
struct AdaptationState {
  static constexpr unsigned kNumValues = 9;

  DualAveragingState dualAveraging;
  WelfordState welford;
  Value windowIdx;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = dualAveraging.toValues();
    llvm::append_range(values, welford.toValues());
    values.push_back(windowIdx);
    return values;
  }
  static AdaptationState fromValues(ArrayRef<Value> values) {
    return {DualAveragingState::fromValues(values.take_front(5)),
            WelfordState::fromValues(values.slice(5, 3)), values[8]};
  }
};

/// What one HMC/NUTS chain carries from one transition to the next.
struct ChainState {
  Value q;
  Value grad;
  Value potentialEnergy;
  Value rng;
};

/// Values carried by the HMC/NUTS warmup loop. `welford` is carried only
/// while the mass matrix is adapted.
struct WarmupCarry {
  ChainState chain;
  Value stepSize;
  Value invMass;
  Value massMatrixSqrt;
  DualAveragingState dualAveraging;
  std::optional<WelfordState> welford;
  Value windowIdx;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = {
        chain.q,  chain.grad, chain.potentialEnergy, chain.rng,
        stepSize, invMass,    massMatrixSqrt};
    llvm::append_range(values, dualAveraging.toValues());
    if (welford)
      llvm::append_range(values, welford->toValues());
    values.push_back(windowIdx);
    return values;
  }
  /// The position, gradient, potential energy and step size are typed by
  /// `positionType` and `scalarType`, the window index by `tensor<i64>`.
  SmallVector<Type> getTypes(RankedTensorType positionType,
                             RankedTensorType scalarType) const {
    SmallVector<Type> types = {positionType,
                               positionType,
                               scalarType,
                               chain.rng.getType(),
                               scalarType,
                               invMass.getType(),
                               massMatrixSqrt.getType()};
    llvm::append_range(types, dualAveraging.getTypes());
    if (welford)
      llvm::append_range(types, welford->getTypes());
    types.push_back(RankedTensorType::get(
        {}, IntegerType::get(positionType.getContext(), 64)));
    return types;
  }
  static WarmupCarry fromValues(ValueRange range, bool hasWelford) {
    SmallVector<Value> values = llvm::to_vector(range);
    WarmupCarry carry;
    carry.chain = {values[0], values[1], values[2], values[3]};
    carry.stepSize = values[4];
    carry.invMass = values[5];
    carry.massMatrixSqrt = values[6];
    carry.dualAveraging =
        DualAveragingState::fromValues(ArrayRef<Value>(values).slice(7, 5));
    if (hasWelford)
      carry.welford =
          WelfordState::fromValues(ArrayRef<Value>(values).slice(12, 3));
    carry.windowIdx = values.back();
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

/// Operands and position layout that the HMC, NUTS and VI lowerings read from
/// an inference op.
struct InferModel {
  /// The traced model and the custom log-density; at most one is set.
  FlatSymbolRefAttr fn;
  FlatSymbolRefAttr logpdfFn;
  SmallVector<Value> fnInputs;
  SmallVector<Type> fnResultTypes;
  Value originalTrace;
  ArrayAttr selection;
  ArrayAttr allAddresses;
  SmallVector<impulse::SupportInfo> supports;
  /// The `logpdf` region of an `impulse.infer_region`.
  Region *logpdfRegion = nullptr;
  DictionaryAttr autodiffAttrs;
  int64_t positionSize = 0;
  /// Width of one collected HMC/NUTS sample; differs from `positionSize` only
  /// for SIMPLEX sites.
  int64_t constrainedPositionSize = 0;
};

/// Emits the HMC/NUTS chains of one `impulse.infer` or `impulse.infer_region`.
/// The constructor reads what all chains share and creates the kernel's
/// constant operands; `emitChains` then emits the chains one after another.
class MCMCChainEmitter {
public:
  template <typename OpT>
  MCMCChainEmitter(OpBuilder &builder, OpT op, InferModel model,
                   bool debugDump);

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
  /// Takes element `chainIdx` of the leading dimension of `value`.
  Value sliceChain(Value value, int64_t chainIdx);

  HMCContext makeHMCContext(Value invMass, Value massMatrixSqrt,
                            Value stepSize) const;
  NUTSContext makeNUTSContext(Value invMass, Value massMatrixSqrt,
                              Value stepSize, Value potentialEnergy) const;
  /// One HMC or NUTS transition from `state`.
  MCMCKernelResult runKernel(const ChainState &state, Value stepSize,
                             Value invMass, Value massMatrixSqrt);

  /// The chain's resumed state, or the initial state computed by InitHMC.
  ChainState initChainState(Value rng, int64_t chainIdx, Value stepSize,
                            Value invMass, Value massMatrixSqrt);
  /// Returns `saved` when set, else a fresh adaptation state for `stepSize`;
  /// a fresh state gets Welford accumulators only if `withWelford`.
  AdaptationState
  loadOrInitAdaptation(Value stepSize, bool withWelford,
                       const std::optional<AdaptationState> &saved);
  /// Emits the warmup loop and returns its results.
  WarmupCarry emitWarmup(const ChainState &start, Value stepSize, Value invMass,
                         Value massMatrixSqrt,
                         const std::optional<AdaptationState> &saved,
                         int64_t chainIdx);
  /// Emits one warmup iteration at global iteration `iteration` and returns
  /// the values it carries to the next one.
  WarmupCarry emitWarmupStep(const WarmupCarry &carry, Value iteration,
                             const WarmupSchedule &schedule);
  /// Emits the end of a middle adaptation window: when `atMiddleWindowEnd`
  /// holds, the mass matrix is re-estimated from `next.welford` and the
  /// adapted accumulators restart. Updates `next` to the selected values.
  void emitWindowEndRestart(WarmupCarry &next, Value atMiddleWindowEnd);
  /// Emits the sampling loop from `state`, leaves the chain's final state in
  /// `state` and returns the collected samples and diagnostics.
  ChainResult emitSampling(ChainState &state, Value stepSize, Value invMass,
                           Value massMatrixSqrt);
  /// The chain's `state_out` values in `state_schema` order.
  SmallVector<Value>
  packStateOut(const ChainState &state, Value stepSize, Value invMass,
               const std::optional<AdaptationState> &adaptation,
               int64_t chainIdx);

  OpBuilder &builder;
  Location loc;
  InferModel model;
  bool debugDump;

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
  RankedTensorType i1TensorType;
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
/// fills in the supports and position sizes.
template <typename OpT> static InferModel readInferModel(OpT op) {
  InferModel model;
  model.logpdfFn = op.getLogpdfFnAttr();
  if (!model.logpdfFn) {
    model.fn = op.getFnAttr();
    model.originalTrace = op.getOriginalTrace();
  }
  model.fnInputs = llvm::to_vector(op.getInputs().drop_front());
  model.selection = op.getSelectionAttr();
  model.allAddresses = op.getAllAddressesAttr();
  model.autodiffAttrs = op.getAutodiffAttrsAttr();
  if constexpr (std::is_same_v<OpT, impulse::InferRegionOp>)
    model.logpdfRegion = &op.getLogpdf();
  return model;
}

template <typename OpT>
MCMCChainEmitter::MCMCChainEmitter(OpBuilder &builder, OpT op, InferModel model,
                                   bool debugDump)
    : builder(builder), loc(op.getLoc()), model(std::move(model)),
      debugDump(debugDump), numWarmup(op.getNumWarmup()),
      totalWarmup(op.getNumWarmup()), numSamples(op.getNumSamples()),
      thinning(op.getThinning()), numChains(op.getNumChains()),
      schema(op.getStateSchemaAttr()), stateIn(op.getStateIn()),
      initPosition(op.getInitPosition()), initStepSize(op.getInitStepSize()),
      initInvMass(op.getInitInverseMassMatrix()),
      isHMC(op.getHmcConfig().has_value()),
      isNUTS(op.getNutsConfig().has_value()) {
  if (auto total = op.getTotalWarmup())
    totalWarmup = *total;

  exposeAdaptation =
      impulse::slotIndex(schema, state_slot::kWindowIdx).has_value();
  if (std::optional<unsigned> first =
          impulse::slotIndex(schema, state_slot::kDaLogStepSize);
      first && *first < stateIn.size())
    savedAdaptation = AdaptationState::fromValues(
        llvm::to_vector(stateIn.slice(*first, AdaptationState::kNumValues)));

  // Multi-chain slots are chain-major; emitChain reads them per chain.
  if (numChains <= 1) {
    if (Value savedStepSize = stateInSlot(state_slot::kStepSize))
      initStepSize = savedStepSize;
    if (Value savedInvMass = stateInSlot(state_slot::kInverseMassMatrix))
      initInvMass = savedInvMass;
  }

  elemType = cast<RankedTensorType>(initStepSize.getType()).getElementType();
  positionType = RankedTensorType::get({1, this->model.positionSize}, elemType);
  scalarType = RankedTensorType::get({}, elemType);
  i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  i1TensorType = RankedTensorType::get({}, builder.getI1Type());

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
    Value chainRng = savedKeys ? sliceChain(savedKeys, chainIdx)
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
      stepSize = sliceChain(savedStepSize, chainIdx);
    if (Value savedInvMass = stateInSlot(state_slot::kInverseMassMatrix))
      invMass = sliceChain(savedInvMass, chainIdx);
    if (saved) {
      SmallVector<Value> values = saved->toValues();
      for (Value &value : values)
        value = sliceChain(value, chainIdx);
      saved = AdaptationState::fromValues(values);
    }
  }

  Value massMatrixSqrt =
      computeMassMatrixSqrt(builder, loc, invMass, positionType);
  ChainState state =
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
    stepSize =
        conditionalDump(builder, loc, warmedUp.stepSize,
                        "MCMC: adapted step size after warmup", debugDump);
    if (adaptMassMatrix) {
      invMass = conditionalDump(
          builder, loc, invMass,
          "MCMC: adapted inverse mass matrix after warmup", debugDump);
    }
    if (exposeAdaptation)
      finalAdaptation = AdaptationState{
          warmedUp.dualAveraging,
          adaptMassMatrix
              ? *warmedUp.welford
              : initWelford(builder, loc, model.positionSize, diagonal),
          warmedUp.windowIdx};
  } else if (exposeAdaptation) {
    finalAdaptation =
        loadOrInitAdaptation(stepSize, /*withWelford=*/true, saved);
  }

  ChainResult result = emitSampling(state, stepSize, invMass, massMatrixSqrt);
  result.trace = conditionalDump(builder, loc, result.trace,
                                 "MCMC: collected samples", debugDump);
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
    value = sliceChain(value, chainIdx);
  return value;
}

Value MCMCChainEmitter::sliceChain(Value value, int64_t chainIdx) {
  auto type = cast<RankedTensorType>(value.getType());
  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<int64_t> start(shape.size(), 0),
      limit(shape.begin(), shape.end()), stride(shape.size(), 1);
  start[0] = chainIdx;
  limit[0] = chainIdx + 1;
  SmallVector<int64_t> oneShape(shape.begin(), shape.end());
  oneShape[0] = 1;
  auto oneType = RankedTensorType::get(oneShape, type.getElementType());
  Value slice = impulse::SliceOp::create(builder, loc, oneType, value,
                                         builder.getDenseI64ArrayAttr(start),
                                         builder.getDenseI64ArrayAttr(limit),
                                         builder.getDenseI64ArrayAttr(stride));
  auto dropType =
      RankedTensorType::get(shape.drop_front(), type.getElementType());
  return impulse::ReshapeOp::create(builder, loc, dropType, slice);
}

HMCContext MCMCChainEmitter::makeHMCContext(Value invMass, Value massMatrixSqrt,
                                            Value stepSize) const {
  HMCContext ctx =
      model.logpdfFn
          ? HMCContext(model.logpdfFn, model.fnInputs, invMass, massMatrixSqrt,
                       stepSize, trajectoryLength, model.positionSize,
                       model.autodiffAttrs)
          : HMCContext(model.fn, model.fnInputs, model.fnResultTypes,
                       model.originalTrace, model.selection, model.allAddresses,
                       invMass, massMatrixSqrt, stepSize, trajectoryLength,
                       model.positionSize, model.supports, model.autodiffAttrs);
  ctx.numSteps = numSteps;
  ctx.logpdfRegion = model.logpdfRegion;
  ctx.debugDump = debugDump;
  return ctx;
}

NUTSContext MCMCChainEmitter::makeNUTSContext(Value invMass,
                                              Value massMatrixSqrt,
                                              Value stepSize,
                                              Value potentialEnergy) const {
  NUTSContext ctx =
      model.logpdfFn
          ? NUTSContext(model.logpdfFn, model.fnInputs, invMass, massMatrixSqrt,
                        stepSize, model.positionSize, potentialEnergy,
                        maxDeltaEnergy, maxTreeDepth, model.autodiffAttrs)
          : NUTSContext(model.fn, model.fnInputs, model.fnResultTypes,
                        model.originalTrace, model.selection,
                        model.allAddresses, invMass, massMatrixSqrt, stepSize,
                        model.positionSize, model.supports, potentialEnergy,
                        maxDeltaEnergy, maxTreeDepth, model.autodiffAttrs);
  ctx.logpdfRegion = model.logpdfRegion;
  ctx.debugDump = debugDump;
  return ctx;
}

MCMCKernelResult MCMCChainEmitter::runKernel(const ChainState &state,
                                             Value stepSize, Value invMass,
                                             Value massMatrixSqrt) {
  if (isHMC)
    return SampleHMC(builder, loc, state.q, state.grad, state.potentialEnergy,
                     state.rng,
                     makeHMCContext(invMass, massMatrixSqrt, stepSize));
  return SampleNUTS(builder, loc, state.q, state.grad, state.potentialEnergy,
                    state.rng,
                    makeNUTSContext(invMass, massMatrixSqrt, stepSize,
                                    state.potentialEnergy));
}

ChainState MCMCChainEmitter::initChainState(Value rng, int64_t chainIdx,
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
      chainInitPosition = sliceChain(chainInitPosition, chainIdx);
  }
  InitialHMCState init = InitHMC(builder, loc, rng, ctx, chainInitPosition);
  return {init.q0, init.grad0, init.U0, init.rng};
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
MCMCChainEmitter::emitWarmup(const ChainState &start, Value stepSize,
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
                      "MCMC: initial step size before warmup", debugDump);
  AdaptationState adaptation =
      loadOrInitAdaptation(initialStepSize, adaptMassMatrix, saved);

  WarmupCarry init;
  init.chain = start;
  init.stepSize = initialStepSize;
  init.invMass = invMass;
  init.massMatrixSqrt = massMatrixSqrt;
  init.dualAveraging = adaptation.dualAveraging;
  if (adaptMassMatrix)
    init.welford = adaptation.welford;
  init.windowIdx = adaptation.windowIdx;
  SmallVector<Type> warmupLoopTypes = init.getTypes(positionType, scalarType);

  auto warmupLoop =
      impulse::ForOp::create(builder, loc, warmupLoopTypes, schedule.zero,
                             numWarmupConst, schedule.one, init.toValues());

  Block *warmupBody = builder.createBlock(&warmupLoop.getRegion());
  warmupBody->addArgument(i64TensorType, loc);
  for (Type type : warmupLoopTypes)
    warmupBody->addArgument(type, loc);

  builder.setInsertionPointToStart(warmupBody);

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
    updatedDaState = updateDualAveraging(builder, loc, carry.dualAveraging,
                                         sample.accept_prob, daConfig);
    currentStepSizeFromDA =
        getStepSizeFromDualAveraging(builder, loc, updatedDaState);
    finalStepSizeFromDA =
        getStepSizeFromDualAveraging(builder, loc, updatedDaState, true);
  } else {
    updatedDaState = carry.dualAveraging;
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

  auto windowIdxGtZero = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sgt, carry.windowIdx, schedule.zero);
  auto windowIdxLtLast =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                            carry.windowIdx, schedule.lastWindow);
  auto isMiddleWindow =
      arith::AndIOp::create(builder, loc, windowIdxGtZero, windowIdxLtLast);

  WelfordState conditionalWelford;
  if (adaptMassMatrix) {
    auto sampleType1D = RankedTensorType::get({model.positionSize}, elemType);
    Value sample1D =
        impulse::ReshapeOp::create(builder, loc, sampleType1D, sample.q);
    const WelfordState &welford = *carry.welford;
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

  Value atWindowEnd = arith::ConstantOp::create(
      builder, loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(false)));

  for (auto [window, windowEnd] : llvm::enumerate(schedule.windowEnds)) {
    auto inWindow = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, carry.windowIdx,
        emitI64ScalarConstant(builder, loc, window));
    auto atEnd = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       iteration, windowEnd);
    auto matchesThisWindow =
        arith::AndIOp::create(builder, loc, inWindow, atEnd);
    atWindowEnd =
        arith::OrIOp::create(builder, loc, atWindowEnd, matchesThisWindow);
  }

  Value newWindowIdx =
      arith::AddIOp::create(builder, loc, carry.windowIdx, schedule.one);
  Value windowIdxAfterIncrement = impulse::SelectOp::create(
      builder, loc, i64TensorType, atWindowEnd, newWindowIdx, carry.windowIdx);

  auto atMiddleWindowEnd =
      arith::AndIOp::create(builder, loc, atWindowEnd, isMiddleWindow);

  WarmupCarry next;
  next.chain = {sample.q, sample.grad, sample.U, sample.rng};
  next.stepSize = adaptedStepSizeInLoop;
  next.invMass = carry.invMass;
  next.massMatrixSqrt = carry.massMatrixSqrt;
  next.dualAveraging = updatedDaState;
  if (adaptMassMatrix)
    next.welford = conditionalWelford;
  next.windowIdx = windowIdxAfterIncrement;

  emitWindowEndRestart(next, atMiddleWindowEnd);
  return next;
}

void MCMCChainEmitter::emitWindowEndRestart(WarmupCarry &next,
                                            Value atMiddleWindowEnd) {
  auto windowEndValues = [](const WarmupCarry &c) {
    SmallVector<Value> values = {c.invMass, c.massMatrixSqrt};
    if (c.welford)
      llvm::append_range(values, c.welford->toValues());
    llvm::append_range(values, c.dualAveraging.toValues());
    return values;
  };
  SmallVector<Value> keptValues = windowEndValues(next);
  auto ifOp = impulse::IfOp::create(
      builder, loc, ValueRange(keptValues).getTypes(), atMiddleWindowEnd);

  builder.setInsertionPointToStart(builder.createBlock(&ifOp.getTrueBranch()));
  WarmupCarry restarted = next;
  if (adaptMassMatrix) {
    Value newInvMass =
        finalizeWelford(builder, loc, *next.welford, welfordConfig);
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
    restarted.welford = initWelford(builder, loc, model.positionSize, diagonal);
  }
  if (adaptStepSize)
    restarted.dualAveraging = initDualAveraging(builder, loc, next.stepSize);
  impulse::YieldOp::create(builder, loc, windowEndValues(restarted));

  builder.setInsertionPointToStart(builder.createBlock(&ifOp.getFalseBranch()));
  impulse::YieldOp::create(builder, loc, keptValues);

  builder.setInsertionPointAfter(ifOp);
  SmallVector<Value> updated = llvm::to_vector_of<Value>(ifOp.getResults());
  next.invMass = updated[0];
  next.massMatrixSqrt = updated[1];
  if (adaptMassMatrix)
    next.welford =
        WelfordState::fromValues(ArrayRef<Value>(updated).slice(2, 3));
  next.dualAveraging =
      DualAveragingState::fromValues(ArrayRef<Value>(updated).take_back(5));
}

ChainResult MCMCChainEmitter::emitSampling(ChainState &state, Value stepSize,
                                           Value invMass,
                                           Value massMatrixSqrt) {
  int64_t collectionSize = numSamples / thinning;
  int64_t startIdx = numSamples % thinning;

  auto samplesBufferType = RankedTensorType::get(
      {collectionSize, model.constrainedPositionSize}, elemType);
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

  Block *loopBody = builder.createBlock(&forLoopOp.getRegion());
  loopBody->addArgument(i64TensorType, loc);
  for (Type type : loopResultTypes)
    loopBody->addArgument(type, loc);

  builder.setInsertionPointToStart(loopBody);
  Value iterIdx = loopBody->getArgument(0);
  ChainState chainLoop = {loopBody->getArgument(1), loopBody->getArgument(2),
                          loopBody->getArgument(3), loopBody->getArgument(4)};
  Value samplesBufferLoop = loopBody->getArgument(5);
  Value diagnosticsBufferLoop = loopBody->getArgument(6);
  Value logDensityBufferLoop = loopBody->getArgument(7);

  auto sample = runKernel(chainLoop, stepSize, invMass, massMatrixSqrt);
  auto qConstrained =
      impulse::constrainPosition(builder, loc, sample.q, model.supports);

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

  auto negU = arith::NegFOp::create(builder, loc, sample.U);
  auto logDensity1D = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({1}, elemType), negU);
  auto updatedLogDensityBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, logDensityBufferType, logDensityBufferLoop, logDensity1D,
      ValueRange{storageIdx});
  auto selectedLogDensityBuffer =
      impulse::SelectOp::create(builder, loc, logDensityBufferType, shouldStore,
                                updatedLogDensityBuffer, logDensityBufferLoop);

  impulse::YieldOp::create(builder, loc,
                           ValueRange{sample.q, sample.grad, sample.U,
                                      sample.rng, selectedSamplesBuffer,
                                      selectedDiagnosticsBuffer,
                                      selectedLogDensityBuffer});

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
    const ChainState &state, Value stepSize, Value invMass,
    const std::optional<AdaptationState> &adaptation, int64_t chainIdx) {
  Value warmupOffsetIn = chainStateInSlot(state_slot::kWarmupOffset, chainIdx);
  if (!warmupOffsetIn)
    warmupOffsetIn = emitI64ScalarConstant(builder, loc, 0);
  Value warmupOffsetOut =
      arith::AddIOp::create(builder, loc, warmupOffsetIn,
                            emitI64ScalarConstant(builder, loc, numWarmup));

  SmallVector<Value> adaptationOut;
  if (adaptation)
    adaptationOut = adaptation->toValues();
  auto slotValue = [&](StringRef name) -> Value {
    if (name == state_slot::kPosition)
      return state.q;
    if (name == state_slot::kGradient)
      return state.grad;
    if (name == state_slot::kPotentialEnergy)
      return state.potentialEnergy;
    if (name == state_slot::kStepSize)
      return stepSize;
    if (name == state_slot::kInverseMassMatrix)
      return invMass;
    if (name == state_slot::kWarmupOffset)
      return warmupOffsetOut;
    unsigned first = *impulse::slotIndex(schema, state_slot::kDaLogStepSize);
    return adaptationOut[*impulse::slotIndex(schema, name) - first];
  };

  // The op verifier checks that `state_schema` has the canonical layout.
  SmallVector<Value> stateOut;
  auto invMassTy = cast<RankedTensorType>(invMass.getType());
  for (const impulse::StateSlot &slot : impulse::buildStateSchema(
           exposeAdaptation, model.positionSize, elemType, invMassTy))
    stateOut.push_back(slotValue(slot.name));
  return stateOut;
}

static ArrayAttr buildSubSelection(OpBuilder &builder, ArrayAttr selection,
                                   Attribute targetSymbol) {
  SmallVector<Attribute> subAddresses;
  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;
    if (address[0] == targetSymbol && address.size() > 1) {
      SmallVector<Attribute> tail(address.begin() + 1, address.end());
      subAddresses.push_back(builder.getArrayAttr(tail));
    }
  }
  return builder.getArrayAttr(subAddresses);
}

namespace {
/// The trace, weight and Hastings correction carried from one sample site to
/// the next while a function body is lowered. `offset` is the next trace
/// column; it advances over both `impulse.if` branches, which take disjoint
/// columns.
struct BranchCarry {
  Value currTrace;
  Value weight;
  int64_t offset;
  Value corr = nullptr;
};
} // namespace

using ProcessSampleFn =
    llvm::function_ref<LogicalResult(impulse::SampleOp, BranchCarry &)>;

// kMHHastingsAttr: weight = [weight, log q(fresh) - log q(replaced)].
// kMHSiteScoresAttr: weight = [liveness; masked log density] per address.
static constexpr llvm::StringLiteral kMHHastingsAttr = "impulse.mh_hastings";
static constexpr llvm::StringLiteral kMHSiteScoresAttr =
    "impulse.mh_site_scores";

/// Returns the element of `tensor` at `index` as a rank-0 tensor.
static Value scalarAt(OpBuilder &b, Location loc, Value tensor,
                      ArrayRef<int64_t> index) {
  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> limit, strides(index.size(), 1), ones(index.size(), 1);
  for (int64_t i : index)
    limit.push_back(i + 1);
  Value slice = impulse::SliceOp::create(
      b, loc, RankedTensorType::get(ones, type.getElementType()), tensor,
      b.getDenseI64ArrayAttr(index), b.getDenseI64ArrayAttr(limit),
      b.getDenseI64ArrayAttr(strides));
  return impulse::ReshapeOp::create(
      b, loc, RankedTensorType::get({}, type.getElementType()), slice);
}

/// Lowers the sample sites of `block` in program order with `processSample`,
/// including those nested in the regions of its ops.
static LogicalResult processImpulseBlock(Block &block,
                                         PatternRewriter &rewriter,
                                         BranchCarry &carry,
                                         ProcessSampleFn processSample);

/// Branches start from the pre-if snapshot; untaken slots stay zero.
static LogicalResult processImpulseIf(impulse::IfOp ifOp,
                                      PatternRewriter &rewriter,
                                      BranchCarry &carry,
                                      ProcessSampleFn processSample) {
  Location loc = ifOp.getLoc();
  BranchCarry before = carry;

  Block &trueBlock = ifOp.getTrueBranch().front();
  if (failed(processImpulseBlock(trueBlock, rewriter, carry, processSample)))
    return failure();
  BranchCarry afterTrue = carry;

  Block &falseBlock = ifOp.getFalseBranch().front();
  carry.currTrace = before.currTrace;
  carry.weight = before.weight;
  carry.corr = before.corr;
  if (failed(processImpulseBlock(falseBlock, rewriter, carry, processSample)))
    return failure();

  auto extendYield = [&](Block &block, const BranchCarry &branch) {
    auto yield = cast<impulse::YieldOp>(block.getTerminator());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yield);
    SmallVector<Value> operands(yield.getOperands().begin(),
                                yield.getOperands().end());
    operands.push_back(branch.currTrace);
    operands.push_back(branch.weight);
    if (branch.corr)
      operands.push_back(branch.corr);
    impulse::YieldOp::create(rewriter, yield.getLoc(), operands);
    rewriter.eraseOp(yield);
  };
  extendYield(trueBlock, afterTrue);
  extendYield(falseBlock, carry);

  SmallVector<Type> newResultTypes(ifOp.getResultTypes().begin(),
                                   ifOp.getResultTypes().end());
  unsigned oldNumResults = ifOp.getNumResults();
  newResultTypes.push_back(before.currTrace.getType());
  // Not always scalar: an MH site-scores regenerate carries a vector.
  newResultTypes.push_back(before.weight.getType());
  if (before.corr)
    newResultTypes.push_back(before.corr.getType());

  rewriter.setInsertionPoint(ifOp);
  auto newIf =
      impulse::IfOp::create(rewriter, loc, newResultTypes, ifOp.getPredicate());
  rewriter.inlineRegionBefore(ifOp.getTrueBranch(), newIf.getTrueBranch(),
                              newIf.getTrueBranch().end());
  rewriter.inlineRegionBefore(ifOp.getFalseBranch(), newIf.getFalseBranch(),
                              newIf.getFalseBranch().end());

  carry.currTrace = newIf.getResult(oldNumResults);
  carry.weight = newIf.getResult(oldNumResults + 1);
  if (before.corr)
    carry.corr = newIf.getResult(oldNumResults + 2);

  rewriter.replaceOp(ifOp, newIf.getResults().take_front(oldNumResults));
  return success();
}

static LogicalResult processImpulseBlock(Block &block,
                                         PatternRewriter &rewriter,
                                         BranchCarry &carry,
                                         ProcessSampleFn processSample) {
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    ops.push_back(&op);

  for (Operation *op : ops) {
    if (auto sampleOp = dyn_cast<impulse::SampleOp>(op)) {
      if (failed(processSample(sampleOp, carry)))
        return failure();
    } else if (auto ifOp = dyn_cast<impulse::IfOp>(op)) {
      bool hasSample = false;
      ifOp->walk([&](impulse::SampleOp) { hasSample = true; });
      if (hasSample &&
          failed(processImpulseIf(ifOp, rewriter, carry, processSample)))
        return failure();
    } else {
      for (Region &region : op->getRegions())
        for (Block &nested : region.getBlocks())
          if (failed(
                  processImpulseBlock(nested, rewriter, carry, processSample)))
            return failure();
    }
  }
  return success();
}

/// Returns entry `index` of `vec` along its leading dimension, with that
/// dimension dropped.
static Value takeLeading(OpBuilder &builder, Location loc, Value vec,
                         int64_t index) {
  auto vecType = cast<RankedTensorType>(vec.getType());
  SmallVector<int64_t> start(vecType.getRank(), 0), limit(vecType.getShape()),
      strides(vecType.getRank(), 1), shape(vecType.getShape());
  start[0] = index;
  limit[0] = index + 1;
  shape[0] = 1;
  Value slice = impulse::SliceOp::create(
      builder, loc, RankedTensorType::get(shape, vecType.getElementType()), vec,
      builder.getDenseI64ArrayAttr(start), builder.getDenseI64ArrayAttr(limit),
      builder.getDenseI64ArrayAttr(strides));
  auto elementType = RankedTensorType::get(vecType.getShape().drop_front(),
                                           vecType.getElementType());
  return impulse::ReshapeOp::create(builder, loc, elementType, slice);
}

/// Packs the rank-0 `scalars` into a new vector of type `vecType`, in order.
static Value packScalars(OpBuilder &builder, Location loc,
                         RankedTensorType vecType, ArrayRef<Value> scalars) {
  auto cellType = RankedTensorType::get({1}, vecType.getElementType());
  Value vec = emitSplatConstant(builder, loc, vecType, 0.0);
  for (auto [index, scalar] : llvm::enumerate(scalars)) {
    Value cell = impulse::ReshapeOp::create(builder, loc, cellType, scalar);
    vec = impulse::DynamicUpdateSliceOp::create(
        builder, loc, vecType, vec, cell,
        ValueRange{emitI64ScalarConstant(builder, loc, index)});
  }
  return vec;
}

/// Writes `update` into the rank-2 `tensor` starting at row 0, column
/// `column`.
static Value updateColumns(OpBuilder &builder, Location loc, Value tensor,
                           Value update, int64_t column) {
  Value row = emitI64ScalarConstant(builder, loc, 0);
  Value col = emitI64ScalarConstant(builder, loc, column);
  return impulse::DynamicUpdateSliceOp::create(
      builder, loc, tensor.getType(), tensor, update, ValueRange{row, col});
}

/// Flattens `value` to a single row and writes it into `trace` starting at
/// column `column`.
static Value writeTraceColumns(OpBuilder &builder, Location loc, Value trace,
                               Value value, int64_t column) {
  auto valueType = cast<RankedTensorType>(value.getType());
  auto rowType = RankedTensorType::get(
      {1, computeTensorElementCount(valueType)}, valueType.getElementType());
  Value row = impulse::ReshapeOp::create(builder, loc, rowType, value);
  return updateColumns(builder, loc, trace, row, column);
}

/// Writes the `values` of `site` into `trace` one after another from column
/// `offset` on, advancing `offset` past them. With `write` unset only `offset`
/// advances. Fails on a dynamically shaped value.
static LogicalResult appendSiteValues(OpBuilder &builder,
                                      impulse::SampleOp site,
                                      ArrayRef<Value> values, Value &trace,
                                      int64_t &offset, bool write = true) {
  for (Value value : values) {
    int64_t numElements =
        computeTensorElementCount(cast<RankedTensorType>(value.getType()));
    if (numElements < 0)
      return site.emitError("Impulse: dynamic tensor dimensions not supported");
    if (write)
      trace = writeTraceColumns(builder, site.getLoc(), trace, value, offset);
    offset += numElements;
  }
  return success();
}

/// Returns the parameters of `site`: its inputs after the rng.
static ValueRange siteParams(impulse::SampleOp site) {
  ValueRange inputs = site.getInputs();
  return inputs.empty() ? inputs : inputs.drop_front();
}

/// Draws fresh values for `site`: inline from its distribution when that
/// sampler can be emitted, else by calling its `fn`. Returns replacements for
/// all results of `site`, the rng first.
static SmallVector<Value> emitSiteDraw(OpBuilder &builder,
                                       SymbolTableCollection &symbolTable,
                                       impulse::SampleOp site) {
  Location loc = site.getLoc();
  if (impulse::DistAttr dist = site.getDistAttr();
      dist && impulse::distSamplerEmittable(dist))
    return impulse::emitDistSampler(
        builder, loc, dist, site.getInputs().front(), siteParams(site),
        cast<RankedTensorType>(site.getResult(1).getType()));
  auto fn = cast<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(site, site.getFnAttr()));
  auto call = func::CallOp::create(builder, loc, fn.getName(),
                                   fn.getResultTypes(), site.getInputs());
  return SmallVector<Value>(call.getResults());
}

/// Checks that the `logpdf_fn` function of `site`, if it has one, takes the
/// `numValues` values of the site followed by its parameters. `call` names
/// the call in the diagnostic.
static LogicalResult verifySiteLogpdf(SymbolTableCollection &symbolTable,
                                      impulse::SampleOp site, size_t numValues,
                                      StringRef call = "logpdf call") {
  if (!site.getLogpdfFnAttr())
    return success();
  auto logpdfFn = cast<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(site, site.getLogpdfFnAttr()));
  if (numValues + siteParams(site).size() == logpdfFn.getNumArguments())
    return success();
  return site.emitError("Impulse: failed to construct ")
         << call << "; logpdf function has wrong number of arguments";
}

/// Emits the log density of `values` (the site's results after the rng) under
/// `site`: from its distribution, or by calling its `logpdf_fn` function on the
/// values followed by the site's parameters.
static Value emitSiteLogpdf(OpBuilder &builder,
                            SymbolTableCollection &symbolTable,
                            impulse::SampleOp site, ValueRange values) {
  Location loc = site.getLoc();
  if (impulse::DistAttr dist = site.getDistAttr())
    return impulse::emitDistLogpdf(builder, loc, dist, values.front(),
                                   siteParams(site));
  auto logpdfFn = cast<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(site, site.getLogpdfFnAttr()));
  SmallVector<Value> operands(values);
  llvm::append_range(operands, siteParams(site));
  return func::CallOp::create(builder, loc, logpdfFn.getName(),
                              logpdfFn.getResultTypes(), operands)
      .getResult(0);
}

/// Reads a value of `type` from row 0 of `trace`, starting at column `column`.
static Value sliceTraceValue(OpBuilder &builder, Location loc, Value trace,
                             int64_t column, RankedTensorType type) {
  int64_t numElements = computeTensorElementCount(type);
  auto rowType = RankedTensorType::get({1, numElements}, type.getElementType());
  Value row = impulse::SliceOp::create(
      builder, loc, rowType, trace, builder.getDenseI64ArrayAttr({0, column}),
      builder.getDenseI64ArrayAttr({1, column + numElements}),
      builder.getDenseI64ArrayAttr({1, 1}));
  return impulse::ReshapeOp::create(builder, loc, type, row);
}

/// Reads consecutive values of `types` from row 0 of `trace`, starting at
/// column `offset`.
static SmallVector<Value> sliceTraceValues(OpBuilder &builder, Location loc,
                                           Value trace, int64_t offset,
                                           TypeRange types) {
  SmallVector<Value> values;
  for (Type type : types) {
    auto tensorType = cast<RankedTensorType>(type);
    values.push_back(sliceTraceValue(builder, loc, trace, offset, tensorType));
    offset += computeTensorElementCount(tensorType);
  }
  return values;
}

/// Returns replacements for the results of `site`: its rng input, then its
/// values read from row 0 of `source` starting at column `offset`. Fails on a
/// dynamically shaped value.
static FailureOr<SmallVector<Value>> readSiteValues(OpBuilder &builder,
                                                    impulse::SampleOp site,
                                                    Value source,
                                                    int64_t offset) {
  SmallVector<Value> values{site.getInputs().front()};
  for (Value result : llvm::drop_begin(site->getResults())) {
    auto type = cast<RankedTensorType>(result.getType());
    int64_t numElements = computeTensorElementCount(type);
    if (numElements < 0) {
      site.emitError("Impulse: dynamic tensor dimensions not supported");
      return failure();
    }
    values.push_back(
        sliceTraceValue(builder, site.getLoc(), source, offset, type));
    offset += numElements;
  }
  return values;
}

/// Returns a `1 x width` trace whose leading columns are `values` and whose
/// following columns hold `siteScores`, one score per column.
static Value appendScoreTail(OpBuilder &builder, Location loc, Value values,
                             ArrayRef<Value> siteScores, int64_t width) {
  auto fullType = RankedTensorType::get({1, width}, builder.getF64Type());
  auto cellType = RankedTensorType::get({1, 1}, builder.getF64Type());
  Value zeroFull = emitSplatConstant(builder, loc, fullType, 0.0);
  Value c0 = emitI64ScalarConstant(builder, loc, 0);
  Value trace = impulse::DynamicUpdateSliceOp::create(
      builder, loc, fullType, zeroFull, values, ValueRange{c0, c0});
  int64_t firstScoreColumn =
      cast<RankedTensorType>(values.getType()).getShape()[1];
  for (auto [i, score] : llvm::enumerate(siteScores)) {
    Value cell = impulse::ReshapeOp::create(builder, loc, cellType, score);
    Value column =
        emitI64ScalarConstant(builder, loc, firstScoreColumn + (int64_t)i);
    trace = impulse::DynamicUpdateSliceOp::create(builder, loc, fullType, trace,
                                                  cell, ValueRange{c0, column});
  }
  return trace;
}

/// Packs `weight` and the Hastings correction `corr` into the weight vector
/// of a Hastings regenerate.
static Value packHastingsWeight(OpBuilder &b, Location loc, Value weight,
                                Value corr) {
  auto vecType = RankedTensorType::get({2}, b.getF64Type());
  auto oneType = RankedTensorType::get({1}, b.getF64Type());
  Value acc = emitSplatConstant(b, loc, vecType, 0.0);
  // Named array: a ValueRange over a braced list would dangle in the loop.
  Value parts[] = {weight, corr};
  for (auto [i, v] : llvm::enumerate(parts)) {
    Value idx = emitI64ScalarConstant(b, loc, i);
    Value flat = impulse::ReshapeOp::create(b, loc, oneType, v);
    acc = impulse::DynamicUpdateSliceOp::create(b, loc, vecType, acc, flat,
                                                ValueRange{idx});
  }
  return acc;
}

/// Splits the weight vector of a Hastings regenerate into the weight and the
/// Hastings correction.
static std::pair<Value, Value> unpackHastingsWeight(OpBuilder &b, Location loc,
                                                    Value packed) {
  return {scalarAt(b, loc, packed, {0}), scalarAt(b, loc, packed, {1})};
}

/// Half-open [start, end) column ranges of a trace.
using ColumnWindows = SmallVector<std::pair<int64_t, int64_t>>;

/// Merges half-open [start, end) windows.
static ColumnWindows mergeColumnWindows(ColumnWindows windows) {
  llvm::sort(windows);
  ColumnWindows merged;
  for (auto w : windows) {
    if (w.first >= w.second)
      continue;
    if (!merged.empty() && w.first <= merged.back().second)
      merged.back().second = std::max(merged.back().second, w.second);
    else
      merged.push_back(w);
  }
  return merged;
}

/// Selects `proposed` over `trace` where `accepted`, window by window, so the
/// columns outside `windows` keep the values of `trace` and stay
/// loop-invariant.
static Value writeBackMHWindows(OpBuilder &b, Location loc, Value trace,
                                Value proposed, Value accepted,
                                ArrayRef<std::pair<int64_t, int64_t>> windows) {
  auto traceType = cast<RankedTensorType>(trace.getType());
  Value out = trace;
  for (const auto &window : windows) {
    int64_t start = window.first, end = window.second;
    auto winType =
        RankedTensorType::get({1, end - start}, traceType.getElementType());
    auto sliceOf = [&](Value v) -> Value {
      return impulse::SliceOp::create(
          b, loc, winType, v, b.getDenseI64ArrayAttr({0, start}),
          b.getDenseI64ArrayAttr({1, end}), b.getDenseI64ArrayAttr({1, 1}));
    };
    Value propWin = sliceOf(proposed);
    Value curWin = sliceOf(trace);
    Value upd =
        impulse::SelectOp::create(b, loc, winType, accepted, propWin, curWin);
    out = updateColumns(b, loc, out, upd, start);
  }
  return out;
}

/// Returns the columns of a `width`-wide trace laid out by `layout` that an MH
/// move regenerating `regenerateAddresses` can change: those of the
/// regenerated sites and of the score tail, and on a trace without score tail
/// also those of sites in branches (their liveness can flip) and sub-calls.
static ColumnWindows mhChangedColumnsFlat(Operation *op, FunctionOpInterface fn,
                                          ArrayAttr layout,
                                          ArrayAttr regenerateAddresses,
                                          int64_t width,
                                          SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t positionSize =
      computePositionSizeForSelection(op, fn, layout, symbolTable);
  bool bare = width == positionSize;
  auto inList = [](ArrayAttr list, ArrayAttr address) {
    return list && llvm::is_contained(list.getValue(), address);
  };
  ColumnWindows windows;
  int64_t offset = 0;
  for (Attribute a : layout) {
    auto address = cast<ArrayAttr>(a);
    int64_t size = computePositionSizeForSelection(
        op, fn, ArrayAttr::get(op->getContext(), {address}), symbolTable);
    impulse::SampleOp site = findSampleBySymbol(sampleMap, address[0]);
    bool isDist = site && (site.getDistAttr() || site.getLogpdfFnAttr());
    bool changed = inList(regenerateAddresses, address);
    // Constrained sites outside branches read the old trace (readsConstraint).
    if (!changed && bare)
      changed = !isDist || address.size() > 1 ||
                site->getParentOfType<impulse::IfOp>();
    if (changed)
      windows.push_back({offset, offset + size});
    offset += size;
  }
  if (width > positionSize)
    windows.push_back({positionSize, width});
  return mergeColumnWindows(windows);
}

namespace {
/// A proposed MH move. `weight` is the new joint density, or its change for a
/// score-tail trace; `corr` is subtracted from the joint ratio to form log
/// alpha unless `logRatio` gives log alpha directly.
struct MHProposal {
  Value trace;
  Value weight;
  Value corr;
  Value rng;
  Value logRatio;
};

/// The MH accept test of a proposal: whether it is accepted, the weight it
/// carries when accepted, and the rng after the uniform draw.
struct MHDecision {
  Value accepted;
  Value acceptWeight;
  Value rng;
};

/// The trace, weight and rng after one MH step, and whether it accepted.
struct MHStep {
  Value trace;
  Value weight;
  Value rng;
  Value accepted;
};
} // namespace

/// Emits the MH accept test of `proposal` against the current `weight`: the
/// proposal is accepted when the log of a uniform draw from its rng falls
/// below log alpha.
static MHDecision emitMHDecision(OpBuilder &builder, Location loc, Value weight,
                                 const MHProposal &proposal, bool deltaWeight) {
  Value jointRatio =
      deltaWeight ? proposal.weight
                  : arith::SubFOp::create(builder, loc, proposal.weight, weight)
                        .getResult();
  Value logAlpha =
      proposal.logRatio
          ? proposal.logRatio
          : arith::SubFOp::create(builder, loc, jointRatio, proposal.corr)
                .getResult();
  Value acceptWeight =
      deltaWeight ? arith::AddFOp::create(builder, loc, weight, proposal.weight)
                        .getResult()
                  : proposal.weight;
  Value rng = proposal.rng;
  Value uniform = emitUniformDraw(builder, loc, rng,
                                  cast<RankedTensorType>(weight.getType()));
  Value logRand = math::LogOp::create(builder, loc, uniform);
  Value accepted = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OLT, logRand, logAlpha);
  return {accepted, acceptWeight, rng};
}

/// Returns whether `op` or an op nested in it has an operand in `values`.
static bool usesAnyOf(Operation *op, const DenseSet<Value> &values) {
  auto inValues = [&](Value v) { return values.contains(v); };
  if (llvm::any_of(op->getOperands(), inValues))
    return true;
  if (op->getNumRegions() == 0)
    return false;
  return op
      ->walk([&](Operation *inner) {
        return llvm::any_of(inner->getOperands(), inValues)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

/// Returns, for each per-site yield of `lpBlock` (terminator operands 1..),
/// whether it depends on `seeds`.
static SmallVector<bool> taintedSiteYields(Block &lpBlock, ValueRange seeds) {
  DenseSet<Value> tainted(seeds.begin(), seeds.end());
  for (Operation &op : lpBlock.without_terminator())
    if (usesAnyOf(&op, tainted))
      tainted.insert(op.result_begin(), op.result_end());
  SmallVector<bool> dirty;
  for (Value v : llvm::drop_begin(lpBlock.getTerminator()->getOperands()))
    dirty.push_back(tainted.contains(v));
  return dirty;
}

namespace {
/// Maps values to the selected sites they depend on, as bit vectors over
/// `numSites`. An `impulse.if` result depends on its predicate only when its
/// two yielded values differ.
struct SiteDependence {
  unsigned numSites;
  DenseMap<Value, llvm::BitVector> sites;

  explicit SiteDependence(unsigned numSites) : numSites(numSites) {}

  llvm::BitVector get(Value v) const {
    auto it = sites.find(v);
    return it == sites.end() ? llvm::BitVector(numSites) : it->second;
  }
  void seed(Value v, unsigned k) {
    llvm::BitVector s(numSites);
    s.set(k);
    sites[v] = s;
  }
  void seed(Value v, const llvm::BitVector &s) {
    if (s.any())
      sites[v] = s;
  }
  void run(Block &block) {
    for (Operation &op : block.without_terminator())
      visit(op);
  }
  void visit(Operation &op) {
    // Sample sites are seeded (selected) or fixed by the trace (unselected).
    if (isa<impulse::SampleRegionOp>(op))
      return;
    if (auto ifOp = dyn_cast<impulse::IfOp>(op)) {
      llvm::BitVector predicateSites = get(ifOp.getPredicate());
      Block &trueBlock = ifOp.getTrueBranch().front();
      Block &falseBlock = ifOp.getFalseBranch().front();
      run(trueBlock);
      run(falseBlock);
      Operation *trueYield = trueBlock.getTerminator();
      Operation *falseYield = falseBlock.getTerminator();
      for (auto [i, result] : llvm::enumerate(ifOp.getResults())) {
        Value trueValue = trueYield->getOperand(i);
        Value falseValue = falseYield->getOperand(i);
        llvm::BitVector resultSites = get(trueValue);
        resultSites |= get(falseValue);
        if (trueValue != falseValue)
          resultSites |= predicateSites;
        if (resultSites.any())
          sites[result] = resultSites;
      }
      return;
    }
    llvm::BitVector s(numSites);
    for (Value v : op.getOperands())
      s |= get(v);
    if (op.getNumRegions() != 0) {
      SetVector<Value> captured;
      getUsedValuesDefinedAbove(op.getRegions(), captured);
      for (Value v : captured)
        s |= get(v);
    }
    if (s.any())
      for (Value r : op.getResults())
        sites[r] = s;
  }
};
} // namespace

/// Returns whether an address in `addresses` starts with the symbol of
/// `sampleOp`.
static bool addressInList(impulse::SampleOp sampleOp, ArrayAttr addresses) {
  for (auto addr : addresses) {
    auto address = cast<ArrayAttr>(addr);
    if (!address.empty() && address[0] == sampleOp.getSymbolAttr())
      return true;
  }
  return false;
}

/// Fails with a diagnostic naming `role` when the first address in
/// `addresses` that starts with the symbol of `sampleOp` is nested.
static LogicalResult verifyTopLevelAddress(impulse::SampleOp sampleOp,
                                           ArrayAttr addresses,
                                           StringRef role) {
  for (Attribute attr : addresses) {
    auto address = cast<ArrayAttr>(attr);
    if (address.empty() || address[0] != sampleOp.getSymbolAttr())
      continue;
    if (address.size() == 1)
      return success();
    return sampleOp.emitError(
               "Impulse: distribution function cannot have composite ")
           << role << " address";
  }
  return success();
}

/// Returns the values in `fn` that regenerating `regenerateAddresses` can
/// change. Taint stops at a site that is not regenerated, whose value comes
/// from the trace; an op with regions is tainted when anything nested in it
/// reads a tainted value.
static DenseSet<Value> computeRegenerateTaint(FunctionOpInterface fn,
                                              ArrayAttr regenerateAddresses) {
  DenseSet<Value> tainted;
  bool changed = true;
  while (changed) {
    changed = false;
    auto taint = [&](auto values) {
      for (Value v : values)
        changed |= tainted.insert(v).second;
    };
    fn.getFunctionBody().walk([&](Operation *op) {
      if (auto sampleOp = dyn_cast<impulse::SampleOp>(op)) {
        if (addressInList(sampleOp, regenerateAddresses))
          taint(llvm::drop_begin(op->getResults()));
        return;
      }
      if (!usesAnyOf(op, tainted))
        return;
      taint(op->getResults());
      for (Region &region : op->getRegions())
        for (Block &block : region)
          taint(block.getArguments());
    });
  }
  return tainted;
}

/// Returns whether every site of `fn` runs exactly once, as a score tail with
/// one column per site requires: no `impulse.if`, and only top-level
/// distribution sites.
static bool deltaWeightFeasible(FunctionOpInterface fn) {
  bool feasible = true;
  fn.getFunctionBody().walk([&](Operation *op) {
    if (isa<impulse::IfOp>(op))
      feasible = false;
    if (auto s = dyn_cast<impulse::SampleOp>(op))
      if ((!s.getDistAttr() && !s.getLogpdfFnAttr()) ||
          s->getParentOp() != fn.getOperation())
        feasible = false;
  });
  return feasible;
}

/// Clones the body of the logpdf block `lpBlock` with its arguments bound to
/// `args` and returns the cloned yields: the joint density, then one density
/// per site.
static SmallVector<Value> cloneLogpdfBody(OpBuilder &builder, Block &lpBlock,
                                          ValueRange args) {
  IRMapping map;
  for (auto [arg, value] : llvm::zip(lpBlock.getArguments(), args))
    map.map(arg, value);
  return cloneBlockBody(builder, lpBlock, map);
}

/// Proposes fresh values for the selected sites of `op` from its sampler,
/// reads the other sites from `trace` and scores the result with the logpdf
/// region. `drawOverride` gives the values of replaced sampler draws.
/// `changedColumns` receives the trace columns the move can change. With
/// `delta`, the weight is the sum of new-minus-old scores over the terms the
/// move can change, else the new joint density; with `withLogRatio` too, a
/// move of several sites on a score-tail trace also returns log alpha.
static MHProposal
proposeRegionMove(OpBuilder &b, Location loc, impulse::InferRegionOp op,
                  const RegionTraceLayout &layout, Value trace, Value rng,
                  bool delta, bool withLogRatio,
                  SmallVectorImpl<std::pair<int64_t, int64_t>> &changedColumns,
                  const DenseMap<Operation *, Value> *drawOverride = nullptr) {
  changedColumns.clear();
  ArrayAttr selection = op.getSelectionAttr();
  Region &sampler = op.getSampling();
  Region &logpdf = op.getLogpdf();
  Block &samplerBlock = sampler.front();
  auto traceType = cast<RankedTensorType>(trace.getType());
  int64_t positionSize = layout.positionSize;

  DenseSet<Attribute> selectedSet(selection.begin(), selection.end());

  IRMapping map;
  map.map(samplerBlock.getArgument(0), rng);
  for (unsigned i = 1, e = samplerBlock.getNumArguments(); i < e; ++i)
    map.map(samplerBlock.getArgument(i), op.getInputs()[i]);

  Value curRng = rng;
  Value newTrace = trace;
  DenseMap<Attribute, SmallVector<Value>> proposedSelected;
  // Program-order ordinal = per-site yield index = score-tail column.
  SmallVector<int64_t> selectedOrdinals;
  int64_t siteOrdinal = -1;

  for (Operation &bodyOp : samplerBlock.without_terminator()) {
    auto site = dyn_cast<impulse::SampleRegionOp>(&bodyOp);
    if (!site) {
      b.clone(bodyOp, map);
      continue;
    }
    ++siteOrdinal;
    Attribute addr = site.getAddressAttr();
    int64_t siteColumn = layout.columnOf.lookup(addr);
    if (selectedSet.contains(addr)) {
      selectedOrdinals.push_back(siteOrdinal);
      changedColumns.push_back(
          {siteColumn, siteColumn + sampleRegionSize(site)});
      Block &sub = site.getSampling().front();
      IRMapping subMap;
      subMap.map(sub.getArgument(0), curRng);
      for (unsigned j = 1, e = site.getInputs().size(); j < e; ++j)
        subMap.map(sub.getArgument(j),
                   map.lookupOrDefault(site.getInputs()[j]));
      for (Operation &subOp : sub.without_terminator()) {
        if (drawOverride) {
          auto it = drawOverride->find(&subOp);
          if (it != drawOverride->end()) {
            subMap.map(subOp.getResult(0),
                       subMap.lookupOrDefault(subOp.getOperand(0)));
            subMap.map(subOp.getResult(1), it->second);
            continue;
          }
        }
        b.clone(subOp, subMap);
      }
      Operation *samplerYield = sub.getTerminator();
      curRng = subMap.lookupOrDefault(samplerYield->getOperand(0));
      map.map(site.getResult(0), curRng);
      SmallVector<Value> values;
      int64_t column = siteColumn;
      for (unsigned k = 1, e = site.getNumResults(); k < e; ++k) {
        Value v = subMap.lookupOrDefault(samplerYield->getOperand(k));
        map.map(site.getResult(k), v);
        values.push_back(v);
        newTrace = writeTraceColumns(b, loc, newTrace, v, column);
        column +=
            computeTensorElementCount(cast<RankedTensorType>(v.getType()));
      }
      proposedSelected[addr] = values;
    } else {
      map.map(site.getResult(0), curRng);
      map.map(site.getResults().drop_front(),
              sliceTraceValues(b, loc, trace, siteColumn,
                               TypeRange(site->getResultTypes()).drop_front()));
    }
  }

  Block &lpBlock = logpdf.front();
  SmallVector<Value> newPositions;
  for (Attribute addr : selection)
    llvm::append_range(newPositions, proposedSelected[addr]);
  SmallVector<Value> newYields = cloneLogpdfBody(b, lpBlock, newPositions);
  Value newWeight = newYields[0];
  SmallVector<bool> dirty =
      delta ? taintedSiteYields(lpBlock, lpBlock.getArguments())
            : SmallVector<bool>(newYields.size() - 1, true);

  // A trace wider than positionSize carries a per-site score tail.
  if (traceType.getShape()[1] > positionSize) {
    Value row0 = emitI64ScalarConstant(b, loc, 0);
    auto cellType = RankedTensorType::get({1, 1}, traceType.getElementType());
    for (unsigned j = 1, e = newYields.size(); j < e; ++j) {
      if (!dirty[j - 1])
        continue;
      Value flat = impulse::ReshapeOp::create(b, loc, cellType, newYields[j]);
      Value col = emitI64ScalarConstant(b, loc, positionSize + (int64_t)j - 1);
      newTrace = impulse::DynamicUpdateSliceOp::create(
                     b, loc, traceType, newTrace, flat, ValueRange{row0, col})
                     .getResult();
    }
  }

  SmallVector<Value> oldYields;
  if (traceType.getShape()[1] == positionSize) {
    SmallVector<Value> oldPositions;
    for (Attribute addr : selection)
      llvm::append_range(
          oldPositions,
          sliceTraceValues(b, loc, trace, layout.columnOf.lookup(addr),
                           ValueRange(proposedSelected[addr]).getTypes()));
    oldYields = cloneLogpdfBody(b, lpBlock, oldPositions);
  }
  Value corr;
  Value logRatio;
  for (int64_t j : selectedOrdinals) {
    Value newScore = newYields[1 + j];
    Value oldScore = traceType.getShape()[1] == positionSize
                         ? oldYields[1 + j]
                         : scalarAt(b, loc, trace, {0, positionSize + j});
    Value change = arith::SubFOp::create(b, loc, newScore, oldScore);
    corr =
        corr ? arith::AddFOp::create(b, loc, corr, change).getResult() : change;
  }

  if (delta) {
    SmallVector<int64_t> dirtyTerms;
    for (auto [j, isDirty] : llvm::enumerate(dirty))
      if (isDirty)
        dirtyTerms.push_back(j);
    // Selected sites' own terms cancel against the proposal density, so log
    // alpha is the sum of the remaining dirty terms.
    bool splitRatio = withLogRatio && selectedOrdinals.size() >= 2 &&
                      traceType.getShape()[1] > positionSize;
    llvm::SmallDenseSet<int64_t> ownTerms(selectedOrdinals.begin(),
                                          selectedOrdinals.end());
    Value ratio;
    Value sum;
    for (int64_t j : dirtyTerms) {
      Value newScore = newYields[1 + j];
      Value oldScore = scalarAt(b, loc, trace, {0, positionSize + (int64_t)j});
      Value change = arith::SubFOp::create(b, loc, newScore, oldScore);
      sum =
          sum ? arith::AddFOp::create(b, loc, sum, change).getResult() : change;
      if (splitRatio && !ownTerms.contains(j))
        ratio = ratio ? arith::AddFOp::create(b, loc, ratio, change).getResult()
                      : change;
    }
    if (splitRatio) {
      if (!ratio) {
        auto scalarType = RankedTensorType::get({}, b.getF64Type());
        ratio = emitSplatConstant(b, loc, scalarType, 0.0);
      }
      logRatio = ratio;
    }
    if (!sum) {
      auto scalarType = RankedTensorType::get({}, b.getF64Type());
      sum = emitSplatConstant(b, loc, scalarType, 0.0);
    }
    newWeight = sum;
  }

  // Only dirty score columns can change; write back just those.
  if (delta) {
    for (auto [j, isDirty] : llvm::enumerate(dirty))
      if (isDirty)
        changedColumns.push_back(
            {positionSize + (int64_t)j, positionSize + (int64_t)j + 1});
  } else if (traceType.getShape()[1] > positionSize) {
    changedColumns.push_back({positionSize, traceType.getShape()[1]});
  }
  changedColumns = mergeColumnWindows(std::move(changedColumns));

  return {newTrace, newWeight, corr, curRng, logRatio};
}

/// Proposes the move of the flat-form MH op `op` with a Hastings
/// `impulse.regenerate`. The weight is the change of the joint density on a
/// score-tail trace, else the new joint density.
static MHProposal proposeFlatMove(OpBuilder &b, Location loc,
                                  impulse::InferOp op,
                                  ArrayRef<Type> fnResultTypes, Value trace,
                                  Value rng) {
  auto weightType = RankedTensorType::get({2}, b.getF64Type());
  SmallVector<Type> regenResultTypes{trace.getType(), weightType};
  for (Type t : fnResultTypes)
    regenResultTypes.push_back(t);
  SmallVector<Value> inputs(op.getInputs().begin(), op.getInputs().end());
  inputs[0] = rng;
  auto regen = impulse::RegenerateOp::create(
      b, loc, regenResultTypes, op.getFnAttr(), inputs, trace,
      op.getConstraint(), /*all_addresses=*/op.getAllAddressesAttr(),
      /*selection=*/op.getSelectionAttr(), op.getConstrainedAddressesAttr(),
      b.getStringAttr(""));
  regen->setAttr(kMHHastingsAttr, b.getUnitAttr());
  auto [weight, corr] = unpackHastingsWeight(b, loc, regen.getWeight());
  return {regen.getNewTrace(), weight, corr, regen.getOutputs()[0],
          /*logRatio=*/Value()};
}

namespace {
/// Batches a run of single-site MH moves. Sound only when each per-site yield
/// depends on at most one run site, no site's parameters depend on another,
/// written columns are disjoint and all proposals are one matching draw.
struct MHRunBatcher {
  struct Move {
    // Op handles are mutable: their accessors are non-const.
    mutable impulse::InferRegionOp op;
    Attribute address;
    mutable impulse::SampleRegionOp site;
    mutable impulse::RandomOp draw;
    ColumnWindows windows; // value columns + dirty score columns
    // Score column of yield j: layout.positionSize + j.
    RegionTraceLayout layout;
    int64_t ordinal = -1; // the site's yield index (program order)
  };

  static std::optional<Move> analyzeMove(impulse::InferRegionOp op) {
    if (!op.getMhConfig().has_value() || op.getSampling().empty() ||
        op.getLogpdf().empty() || op.getNumSamples() > 1 ||
        op.getStateIn().size() != 1 || op.getInputs().empty() ||
        op.getConstraint() || !op.getOriginalTrace())
      return std::nullopt;
    ArrayAttr selection = op.getSelectionAttr();
    ArrayAttr allAddresses = op.getAllAddressesAttr();
    if (!selection || selection.size() != 1 || !allAddresses)
      return std::nullopt;
    Block &lpBlock = op.getLogpdf().front();
    SmallVector<impulse::SampleRegionOp> sites = llvm::to_vector(
        op.getSampling().front().getOps<impulse::SampleRegionOp>());
    Operation *lpTerm = lpBlock.getTerminator();
    if (lpTerm->getNumOperands() != 1 + sites.size() ||
        lpBlock.getNumArguments() != 1)
      return std::nullopt;
    FailureOr<RegionTraceLayout> layout =
        getRegionTraceLayout(op, /*topLevelOnly=*/true);
    if (failed(layout))
      return std::nullopt;
    Attribute address = selection[0];
    impulse::SampleRegionOp site = layout->siteByAddress.lookup(address);
    if (!site || site.getNumResults() != 2)
      return std::nullopt;
    int64_t positionSize = layout->positionSize;
    auto traceType = cast<RankedTensorType>(op.getOriginalTrace().getType());
    if (traceType.getRank() != 2 || traceType.getShape()[0] != 1 ||
        traceType.getShape()[1] <= positionSize)
      return std::nullopt;
    Block &sub = site.getSampling().front();
    if (sub.getNumArguments() == 0)
      return std::nullopt;
    impulse::RandomOp draw;
    for (Operation &o : sub.without_terminator()) {
      if (o.getNumRegions() != 0)
        return std::nullopt;
      if (auto r = dyn_cast<impulse::RandomOp>(&o)) {
        if (draw)
          return std::nullopt;
        draw = r;
      }
    }
    if (!draw || draw.getRngState() != sub.getArgument(0) ||
        !sub.getArgument(0).hasOneUse() ||
        !draw.getOutputRngState().hasOneUse() ||
        sub.getTerminator()->getOperand(0) != draw.getOutputRngState())
      return std::nullopt;
    auto dist = draw.getRngDistribution();
    if (dist != impulse::RngDistribution::UNIFORM &&
        dist != impulse::RngDistribution::NORMAL)
      return std::nullopt;
    // Parameters must dominate the batched draw, which precedes every move.
    auto definedOutside = [&](Value v) {
      return !op->isAncestor(v.getParentRegion()->getParentOp());
    };
    if (!isa<RankedTensorType>(draw.getResult().getType()) ||
        !definedOutside(draw.getA()) || !definedOutside(draw.getB()))
      return std::nullopt;
    Move m;
    m.op = op;
    m.address = address;
    m.site = site;
    m.draw = draw;
    m.ordinal = llvm::find(sites, site) - sites.begin();
    int64_t column = layout->columnOf.lookup(address);
    m.windows.push_back({column, column + sampleRegionSize(site)});
    SmallVector<bool> dirty =
        taintedSiteYields(lpBlock, lpBlock.getArguments());
    for (auto [j, d] : llvm::enumerate(dirty))
      if (d)
        m.windows.push_back(
            {positionSize + (int64_t)j, positionSize + (int64_t)j + 1});
    m.windows = mergeColumnWindows(m.windows);
    m.layout = std::move(*layout);
    return m;
  }

  static bool chained(const Move &prev, const Move &next) {
    impulse::InferRegionOp p = prev.op, n = next.op;
    if (p->getBlock() != n->getBlock() || !p->isBeforeInBlock(n))
      return false;
    if (n.getOriginalTrace() != p.getTrace() ||
        n.getInputs()[0] != p.getOutputRngState() ||
        n.getStateIn()[0] != p.getStateOut()[0])
      return false;
    if (n.getInputs().size() != p.getInputs().size() ||
        n.getAllAddressesAttr() != p.getAllAddressesAttr() ||
        n.getOriginalTrace().getType() != p.getOriginalTrace().getType())
      return false;
    for (unsigned i = 1, e = n.getInputs().size(); i < e; ++i)
      if (n.getInputs()[i] != p.getInputs()[i])
        return false;
    return p.getOutputRngState().hasOneUse() && p.getStateOut()[0].hasOneUse();
  }

  static bool sameValueOrConstant(Value a, Value b) {
    if (a == b)
      return true;
    Attribute ca, cb;
    return matchPattern(a, m_Constant(&ca)) &&
           matchPattern(b, m_Constant(&cb)) && ca == cb &&
           a.getType() == b.getType();
  }

  /// Intermediate traces are those of every move but the last.
  struct Between {
    bool ok = true;
    DenseMap<Value, unsigned> intermediate; // trace of move i -> i
    SmallVector<Operation *> derived;       // program order
    DenseSet<Operation *> derivedSet;
    DenseMap<Value, llvm::BitVector> outer; // run-site dependence
    // Used after the run; recomputed on the final trace when finalSafe.
    SetVector<Value> escaping;
  };

  static Between analyzeBetween(ArrayRef<Move> moves) {
    Between between;
    unsigned numMoves = moves.size();
    Block *block = moves.front().op->getBlock();
    Operation *first = moves.front().op, *last = moves.back().op;
    DenseSet<Operation *> moveOps;
    for (const Move &m : moves)
      moveOps.insert(m.op);
    for (unsigned i = 0; i + 1 < numMoves; ++i)
      between.intermediate[moves[i].op.getTrace()] = i;
    auto writtenBy = [&](int64_t c, unsigned from, unsigned to) {
      llvm::BitVector s(numMoves);
      for (unsigned i = from; i < to; ++i)
        for (auto [windowStart, windowEnd] : moves[i].windows)
          if (c >= windowStart && c < windowEnd)
            s.set(i);
      return s;
    };
    // Same value on the final trace as on the intermediate traces it reads.
    DenseMap<Operation *, bool> finalSafe;
    for (Operation *o = first->getNextNode(); o && o != last;
         o = o->getNextNode()) {
      if (moveOps.contains(o))
        continue;
      bool fromIntermediate = false, safe = true;
      llvm::BitVector s(numMoves);
      auto addValue = [&](Value v) {
        if (between.intermediate.count(v)) {
          fromIntermediate = true;
          safe = false;
          s.set();
        } else if (Operation *d = v.getDefiningOp();
                   d && between.derivedSet.contains(d)) {
          fromIntermediate = true;
          safe &= finalSafe.lookup(d);
          s |= between.outer.lookup(v);
        }
      };
      auto slice = dyn_cast<impulse::SliceOp>(o);
      if (slice && between.intermediate.count(slice.getOperand())) {
        ArrayRef<int64_t> starts = slice.getStartIndices(),
                          limits = slice.getLimitIndices(),
                          strides = slice.getStrides();
        if (starts.size() != 2) {
          between.ok = false;
          return between;
        }
        unsigned i = between.intermediate.lookup(slice.getOperand());
        fromIntermediate = true;
        for (int64_t c = starts[1]; c < limits[1]; c += strides[1]) {
          s |= writtenBy(c, 0, numMoves);
          safe &= writtenBy(c, i + 1, numMoves).none();
        }
      } else {
        for (Value v : o->getOperands())
          addValue(v);
        if (o->getNumRegions() != 0) {
          SetVector<Value> captured;
          getUsedValuesDefinedAbove(o->getRegions(), captured);
          for (Value v : captured)
            addValue(v);
        }
      }
      if (!fromIntermediate)
        continue;
      if (!isMemoryEffectFree(o)) {
        between.ok = false;
        return between;
      }
      between.derived.push_back(o);
      between.derivedSet.insert(o);
      finalSafe[o] = safe;
      for (Value r : o->getResults())
        between.outer[r] = s;
    }
    for (unsigned i = 0; i + 1 < numMoves; ++i)
      for (OpOperand &u : moves[i].op.getTrace().getUses()) {
        if (u.getOwner() == moves[i + 1].op.getOperation() &&
            u.getOperandNumber() ==
                moves[i + 1].op.getOriginalTraceMutable()[0].getOperandNumber())
          continue;
        Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
        if (!anc || !between.derivedSet.contains(anc)) {
          between.ok = false;
          return between;
        }
      }
    for (Operation *o : between.derived)
      for (Value r : o->getResults())
        for (OpOperand &u : r.getUses()) {
          Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
          if (!anc || (moveOps.contains(anc) && anc == u.getOwner())) {
            between.ok = false; // an operand of a move, not a region capture
            return between;
          }
          if (last->isBeforeInBlock(anc)) {
            if (!finalSafe.lookup(o)) {
              between.ok = false;
              return between;
            }
            between.escaping.insert(r);
          }
        }
    return between;
  }

  static bool batchable(ArrayRef<Move> moves) {
    unsigned numMoves = moves.size();
    Block *block = moves.front().op->getBlock();
    Operation *last = moves.back().op;
    impulse::RandomOp firstDraw = moves.front().draw;
    for (const Move &m : moves) {
      if (m.draw.getRngDistribution() != firstDraw.getRngDistribution() ||
          m.draw.getResult().getType() != firstDraw.getResult().getType() ||
          !sameValueOrConstant(m.draw.getA(), firstDraw.getA()) ||
          !sameValueOrConstant(m.draw.getB(), firstDraw.getB()))
        return false;
    }
    ColumnWindows all;
    for (const Move &m : moves)
      for (auto w : m.windows)
        all.push_back(w);
    llvm::sort(all);
    for (unsigned i = 1; i < all.size(); ++i)
      if (all[i].first < all[i - 1].second)
        return false;
    for (const Move &m : moves)
      for (Value v : {m.op.getDiagnostics(), m.op.getLogDensities()})
        for (OpOperand &u : v.getUses()) {
          Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
          if (!anc || !last->isBeforeInBlock(anc))
            return false;
        }
    Between between = analyzeBetween(moves);
    if (!between.ok)
      return false;
    DenseMap<Value, unsigned> &intermediate = between.intermediate;
    DenseMap<Value, llvm::BitVector> &outer = between.outer;
    for (unsigned k = 0; k < numMoves; ++k) {
      impulse::InferRegionOp op = moves[k].op;
      auto seedCaptured = [&](SiteDependence &dep, Region &r) {
        SetVector<Value> captured;
        getUsedValuesDefinedAbove(r, captured);
        for (Value v : captured) {
          if (intermediate.count(v))
            return false;
          auto it = outer.find(v);
          if (it != outer.end())
            dep.seed(v, it->second);
        }
        return true;
      };
      SiteDependence lpDep(numMoves);
      if (!seedCaptured(lpDep, op.getLogpdf()))
        return false;
      Block &lpBlock = op.getLogpdf().front();
      lpDep.seed(lpBlock.getArgument(0), k);
      lpDep.run(lpBlock);
      for (Value y : llvm::drop_begin(lpBlock.getTerminator()->getOperands()))
        if (lpDep.get(y).count() > 1)
          return false;
      SiteDependence sDep(numMoves);
      if (!seedCaptured(sDep, op.getSampling()))
        return false;
      Block &sBlock = op.getSampling().front();
      for (Operation &o : sBlock)
        if (auto s = dyn_cast<impulse::SampleRegionOp>(&o))
          for (unsigned j = 0; j < numMoves; ++j)
            if (s.getAddressAttr() == moves[j].address)
              for (Value r : llvm::drop_begin(s.getResults()))
                sDep.seed(r, j);
      sDep.run(sBlock);
      for (Value param : llvm::drop_begin(moves[k].site->getOperands())) {
        llvm::BitVector p = sDep.get(param);
        p.reset(k);
        if (p.any())
          return false;
      }
    }
    return true;
  }

  static void emit(ArrayRef<Move> moves) {
    unsigned numMoves = moves.size();
    impulse::InferRegionOp first = moves.front().op, last = moves.back().op;
    Location loc = last.getLoc();
    Value trace0 = first.getOriginalTrace();
    Value weight0 = first.getStateIn()[0];
    Value rng0 = first.getInputs()[0];
    auto traceType = cast<RankedTensorType>(trace0.getType());
    auto weightType = cast<RankedTensorType>(weight0.getType());
    Type rngType = rng0.getType();
    Between between = analyzeBetween(moves);
    DenseMap<Operation *, SmallVector<unsigned>> readsIntermediate;
    for (Operation *o : between.derived)
      for (OpOperand &u : o->getOpOperands())
        if (between.intermediate.count(u.get()))
          readsIntermediate[o].push_back(u.getOperandNumber());
    for (unsigned i = 0; i + 1 < numMoves; ++i)
      moves[i].op.getTrace().replaceUsesWithIf(trace0, [&](OpOperand &u) {
        return !(u.getOwner() == moves[i + 1].op.getOperation() &&
                 u.getOperandNumber() == moves[i + 1]
                                             .op.getOriginalTraceMutable()[0]
                                             .getOperandNumber());
      });

    OpBuilder b(last);

    impulse::RandomOp firstDraw = moves.front().draw;
    auto drawType = cast<RankedTensorType>(firstDraw.getResult().getType());
    SmallVector<int64_t> batchedShape{(int64_t)numMoves};
    batchedShape.append(drawType.getShape().begin(), drawType.getShape().end());
    auto proposals = impulse::RandomOp::create(
        b, loc,
        TypeRange{rngType, RankedTensorType::get(batchedShape,
                                                 drawType.getElementType())},
        rng0, firstDraw.getA(), firstDraw.getB(),
        firstDraw.getRngDistributionAttr());
    auto vecType =
        RankedTensorType::get({(int64_t)numMoves}, weightType.getElementType());
    Value rngOut = proposals.getOutputRngState();
    Value uniform = emitUniformDraw(b, loc, rngOut, vecType);
    Value logRand = math::LogOp::create(b, loc, uniform);

    SmallVector<Value> newTraces, deltaCells, corrCells;
    SmallVector<ColumnWindows> changed(numMoves);
    auto elemType = traceType.getElementType();
    auto rowType = RankedTensorType::get({1, (int64_t)numMoves}, elemType);
    auto colType = RankedTensorType::get({1, 1}, elemType);
    auto unitSlice = [&](Value t, int64_t c) -> Value {
      return impulse::SliceOp::create(
          b, loc, colType, t, b.getDenseI64ArrayAttr({0, c}),
          b.getDenseI64ArrayAttr({1, c + 1}), b.getDenseI64ArrayAttr({1, 1}));
    };
    for (unsigned k = 0; k < numMoves; ++k) {
      DenseMap<Operation *, Value> drawOverride;
      drawOverride[moves[k].draw] =
          takeLeading(b, loc, proposals.getResult(), k);
      MHProposal proposal = proposeRegionMove(
          b, loc, moves[k].op, moves[k].layout, trace0, rng0, /*delta=*/true,
          /*withLogRatio=*/false, changed[k], &drawOverride);
      newTraces.push_back(proposal.trace);
      deltaCells.push_back(proposal.weight);
      corrCells.push_back(proposal.corr);
    }

    SmallVector<SmallVector<int64_t>> cols(numMoves);
    for (unsigned k = 0; k < numMoves; ++k)
      for (auto [windowStart, windowEnd] : changed[k])
        for (int64_t c = windowStart; c < windowEnd; ++c)
          cols[k].push_back(c);
    // Slotted: all moves write the same column kinds, own score at one slot.
    size_t numCols = cols.front().size();
    int64_t ownSlot = -1;
    bool slotted = true;
    for (unsigned k = 0; k < numMoves && slotted; ++k) {
      slotted = cols[k].size() == numCols;
      for (size_t s = 0; s < numCols && slotted; ++s)
        slotted = (cols[k][s] >= moves[k].layout.positionSize) ==
                  (cols[0][s] >= moves[0].layout.positionSize);
      int64_t own = -1;
      for (size_t s = 0; s < numCols && slotted; ++s)
        if (cols[k][s] == moves[k].layout.positionSize + moves[k].ordinal)
          own = s;
      slotted = slotted && own >= 0 && (k == 0 || own == ownSlot);
      ownSlot = own;
    }
    SmallVector<Value> newOf(numCols), oldOf(numCols);
    Value deltaVec, logAlphaVec;
    if (slotted) {
      for (size_t s = 0; s < numCols; ++s) {
        SmallVector<Value> proposed;
        for (unsigned k = 0; k < numMoves; ++k)
          proposed.push_back(unitSlice(newTraces[k], cols[k][s]));
        newOf[s] = packScalars(b, loc, vecType, proposed);
        int64_t stride = cols[1][s] - cols[0][s];
        bool even = stride >= 1;
        for (unsigned k = 1; k < numMoves && even; ++k)
          even = cols[k][s] - cols[k - 1][s] == stride;
        if (even) {
          oldOf[s] = impulse::ReshapeOp::create(
              b, loc, vecType,
              impulse::SliceOp::create(
                  b, loc, rowType, trace0,
                  b.getDenseI64ArrayAttr({0, cols[0][s]}),
                  b.getDenseI64ArrayAttr({1, cols[numMoves - 1][s] + 1}),
                  b.getDenseI64ArrayAttr({1, stride})));
        } else {
          SmallVector<Value> incoming;
          for (unsigned k = 0; k < numMoves; ++k)
            incoming.push_back(unitSlice(trace0, cols[k][s]));
          oldOf[s] = packScalars(b, loc, vecType, incoming);
        }
      }
      for (size_t s = 0; s < numCols; ++s) {
        if (cols[0][s] < moves[0].layout.positionSize)
          continue;
        Value d = arith::SubFOp::create(b, loc, newOf[s], oldOf[s]);
        deltaVec = deltaVec
                       ? arith::AddFOp::create(b, loc, deltaVec, d).getResult()
                       : d;
      }
      Value corrVec =
          arith::SubFOp::create(b, loc, newOf[ownSlot], oldOf[ownSlot]);
      logAlphaVec = arith::SubFOp::create(b, loc, deltaVec, corrVec);
    } else {
      deltaVec = packScalars(b, loc, vecType, deltaCells);
      logAlphaVec = arith::SubFOp::create(
          b, loc, deltaVec, packScalars(b, loc, vecType, corrCells));
    }
    Value accepted = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                           logRand, logAlphaVec);
    Value trace = trace0, weight = weight0;
    SmallVector<Value> acceptedOf;
    // In sweep order, matching the sequential weight updates.
    for (unsigned k = 0; k < numMoves; ++k) {
      Value acc = takeLeading(b, loc, accepted, k);
      acceptedOf.push_back(acc);
      Value acceptWeight = arith::AddFOp::create(
          b, loc, weight, takeLeading(b, loc, deltaVec, k));
      weight = arith::SelectOp::create(b, loc, acc, acceptWeight, weight);
    }

    if (!slotted) {
      for (unsigned k = 0; k < numMoves; ++k)
        trace = writeBackMHWindows(b, loc, trace, newTraces[k], acceptedOf[k],
                                   changed[k]);
    } else {
      SmallVector<Value> selOf(numCols);
      for (size_t s = 0; s < numCols; ++s)
        selOf[s] =
            arith::SelectOp::create(b, loc, accepted, newOf[s], oldOf[s]);
      // Slots within moveStride columns share a window, viewed as numMoves
      // rows of moveStride columns.
      int64_t moveStride = cols[1][0] - cols[0][0];
      bool strided = moveStride >= 1;
      for (size_t s = 0; s < numCols && strided; ++s)
        for (unsigned k = 0; k < numMoves && strided; ++k)
          strided = cols[k][s] == cols[0][s] + (int64_t)k * moveStride;
      SmallVector<std::pair<int64_t, SmallVector<size_t>>> groups;
      if (strided) {
        SmallVector<size_t> order(numCols);
        for (size_t i = 0; i < numCols; ++i)
          order[i] = i;
        llvm::sort(order,
                   [&](size_t x, size_t y) { return cols[0][x] < cols[0][y]; });
        for (size_t s : order) {
          if (!groups.empty() && cols[0][s] < groups.back().first + moveStride)
            groups.back().second.push_back(s);
          else
            groups.push_back({cols[0][s], {s}});
        }
        for (size_t i = 0; i < groups.size() && strided; ++i) {
          int64_t end = groups[i].first + (int64_t)numMoves * moveStride;
          strided = end <= traceType.getShape()[1] &&
                    (i + 1 == groups.size() || end <= groups[i + 1].first);
        }
      }
      if (strided) {
        auto blockType =
            RankedTensorType::get({(int64_t)numMoves, moveStride}, elemType);
        auto blockColType =
            RankedTensorType::get({(int64_t)numMoves, 1}, elemType);
        auto winType = RankedTensorType::get(
            {1, (int64_t)numMoves * moveStride}, elemType);
        for (auto &[g, slots] : groups) {
          Value block = impulse::ReshapeOp::create(
              b, loc, blockType,
              impulse::SliceOp::create(
                  b, loc, winType, trace0, b.getDenseI64ArrayAttr({0, g}),
                  b.getDenseI64ArrayAttr(
                      {1, g + (int64_t)numMoves * moveStride}),
                  b.getDenseI64ArrayAttr({1, 1})));
          for (size_t s : slots) {
            Value slotCol =
                impulse::ReshapeOp::create(b, loc, blockColType, selOf[s]);
            block = updateColumns(b, loc, block, slotCol, cols[0][s] - g);
          }
          Value blockWin = impulse::ReshapeOp::create(b, loc, winType, block);
          trace = updateColumns(b, loc, trace, blockWin, g);
        }
      } else {
        for (size_t s = 0; s < numCols; ++s) {
          Value row = impulse::ReshapeOp::create(b, loc, rowType, selOf[s]);
          for (unsigned k = 0; k < numMoves; ++k) {
            Value unitCell = unitSlice(row, k);
            trace = updateColumns(b, loc, trace, unitCell, cols[k][s]);
          }
        }
      }
    }

    if (!between.escaping.empty()) {
      IRMapping cloneMap;
      for (Operation *o : between.derived) {
        Operation *c = b.clone(*o, cloneMap);
        for (unsigned i : readsIntermediate.lookup(o))
          c->setOperand(i, trace);
      }
      for (Value v : between.escaping) {
        Value clonedValue = cloneMap.lookup(v);
        v.replaceUsesWithIf(clonedValue, [&](OpOperand &u) {
          Operation *anc =
              last->getBlock()->findAncestorOpInBlock(*u.getOwner());
          return anc && last->isBeforeInBlock(anc);
        });
      }
    }

    for (unsigned k = 0; k < numMoves; ++k) {
      impulse::InferRegionOp op = moves[k].op;
      op.getDiagnostics().replaceAllUsesWith(acceptedOf[k]);
      auto ldType = cast<ShapedType>(op.getLogDensities().getType());
      if (!op.getLogDensities().use_empty())
        op.getLogDensities().replaceAllUsesWith(
            arith::ConstantOp::create(b, loc, b.getZeroAttr(ldType)));
    }
    last.getTrace().replaceAllUsesWith(trace);
    last.getOutputRngState().replaceAllUsesWith(rngOut);
    last.getStateOut()[0].replaceAllUsesWith(weight);
    for (unsigned k = numMoves; k-- > 0;)
      moves[k].op->erase();
  }

  static void run(Operation *root) {
    SmallVector<SmallVector<Move>> chains;
    root->walk([&](Block *block) {
      SmallVector<Move> chain;
      for (Operation &o : *block) {
        auto op = dyn_cast<impulse::InferRegionOp>(&o);
        if (!op)
          continue;
        std::optional<Move> m = analyzeMove(op);
        if (!m) {
          continue;
        }
        if (!chain.empty() && !chained(chain.back(), *m)) {
          if (chain.size() >= 2)
            chains.push_back(chain);
          chain.clear();
        }
        chain.push_back(*m);
      }
      if (chain.size() >= 2)
        chains.push_back(chain);
    });
    // Longest segment first: a shorter one can fail where a longer one holds.
    constexpr size_t kMaxSegment = 256;
    SmallVector<SmallVector<Move>> segments;
    for (auto &chain : chains) {
      size_t i = 0;
      while (i < chain.size()) {
        size_t j = std::min(chain.size(), i + kMaxSegment) - 1;
        for (; j > i; --j)
          if (batchable(ArrayRef<Move>(chain).slice(i, j + 1 - i)))
            break;
        if (j > i)
          segments.push_back(
              SmallVector<Move>(chain.begin() + i, chain.begin() + j + 1));
        i = j + 1;
      }
    }
    for (auto &seg : segments)
      emit(seg);
  }
};
} // namespace

namespace {

struct ExpandImpulsePass
    : public enzyme::impl::ExpandImpulsePassBase<ExpandImpulsePass> {
  using ExpandImpulsePassBase::ExpandImpulsePassBase;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::OpPassManager pm;
    mlir::LogicalResult result = mlir::parsePassPipeline(postpasses, pm);
    if (!mlir::failed(result)) {
      pm.getDependentDialects(registry);
    }

    registry
        .insert<mlir::arith::ArithDialect, mlir::math::MathDialect,
                mlir::complex::ComplexDialect, mlir::cf::ControlFlowDialect,
                mlir::enzyme::EnzymeDialect, mlir::impulse::ImpulseDialect>();
  }

  struct LowerUntracedCallPattern
      : public mlir::OpRewritePattern<impulse::UntracedCallOp> {
    using mlir::OpRewritePattern<impulse::UntracedCallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::UntracedCallOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError("Impulse: trying to call an empty function");
        return failure();
      }

      std::unique_ptr<ImpulseUtils> putils(
          ImpulseUtils::CreateFromClone(fn, ImpulseMode::Call));
      FunctionOpInterface NewF = putils->newFunc;

      SmallVector<Operation *, 4> toErase;
      NewF.walk([&](impulse::SampleOp sampleOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        rewriter.replaceAllUsesWith(
            sampleOp->getResults(),
            emitSiteDraw(rewriter, symbolTable, sampleOp));
        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      rewriter.setInsertionPoint(CI);
      auto newCI =
          func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                               NewF.getResultTypes(), CI.getOperands());

      rewriter.replaceOp(CI, newCI.getResults());

      return success();
    }
  };

  struct LowerSimulatePattern
      : public mlir::OpRewritePattern<impulse::SimulateOp> {
    using mlir::OpRewritePattern<impulse::SimulateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::SimulateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "Impulse: calling `simulate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "`logpdf_fn` attribute to avoid recursive `simulate` calls "
            "which is intended for generative functions");
        return failure();
      }

      ArrayAttr allAddresses = CI.getAllAddressesAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, allAddresses, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("Impulse: failed to compute position size for simulate");
        return failure();
      }

      int64_t declaredWidth =
          cast<RankedTensorType>(CI.getTrace().getType()).getShape()[1];
      int64_t scoreTailSize = declaredWidth - positionSize;
      SmallVector<Value> siteScores;
      std::unique_ptr<ImpulseUtils> putils(ImpulseUtils::CreateFromClone(
          fn, ImpulseMode::Simulate, declaredWidth));
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();
      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      Value zeroWeight =
          emitSplatConstant(entryBuilder, initLoc, scalarType, 0.0);

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      Value zeroTrace =
          emitSplatConstant(entryBuilder, initLoc, traceType, 0.0);

      SmallVector<Operation *> toErase;
      BranchCarry carry{zeroTrace, zeroWeight, /*offset=*/0};
      auto processSample = [&](impulse::SampleOp sampleOp,
                               BranchCarry &branch) -> LogicalResult {
        Value &currTrace = branch.currTrace;
        Value &weightAccumulator = branch.weight;
        int64_t &currentOffset = branch.offset;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        if (sampleOp.getDistAttr() || sampleOp.getLogpdfFnAttr()) {
          sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          ArrayRef<Value> values = ArrayRef<Value>(sampledValues).drop_front();
          if (failed(verifySiteLogpdf(symbolTable, sampleOp, values.size())))
            return failure();
          Value logpdf =
              emitSiteLogpdf(rewriter, symbolTable, sampleOp, values);
          weightAccumulator = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                    weightAccumulator, logpdf);
          if (scoreTailSize > 0)
            siteScores.push_back(logpdf);
          if (addressInList(sampleOp, allAddresses) &&
              failed(appendSiteValues(rewriter, sampleOp, values, currTrace,
                                      currentOffset)))
            return failure();
        } else {
          if (scoreTailSize > 0) {
            sampleOp.emitError("Impulse: score-tail traces do not support "
                               "nested generative sites");
            return failure();
          }
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "Impulse: generative function body is empty; "
                "if this is a distribution, add a `logpdf_fn` attribute");
            return failure();
          }

          ArrayAttr subAllAddresses = buildSubSelection(
              rewriter, allAddresses, sampleOp.getSymbolAttr());
          if (subAllAddresses.empty()) {
            sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subAllAddresses, symbolTable);
            if (subPositionSize <= 0) {
              sampleOp.emitError(
                  "Impulse: failed to compute sub-position size");
              return failure();
            }

            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> simResultTypes;
            simResultTypes.push_back(subTraceType);
            simResultTypes.push_back(scalarTy);
            for (auto t : genFn.getResultTypes())
              simResultTypes.push_back(t);

            auto nestedSimulate = impulse::SimulateOp::create(
                rewriter, sampleOp.getLoc(), simResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subAllAddresses);
            auto subTrace = nestedSimulate.getTrace();
            auto subWeight = nestedSimulate.getWeight();

            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);

            int64_t mergeOffset = computeOffsetForSampleInSelection(
                sampleOp, fn, allAddresses, sampleOp.getSymbolAttr(),
                symbolTable);
            if (mergeOffset < 0) {
              sampleOp.emitError("Impulse: failed to compute merge offset");
              return failure();
            }

            currTrace = updateColumns(rewriter, sampleOp.getLoc(), currTrace,
                                      subTrace, mergeOffset);
            currentOffset =
                std::max(currentOffset, mergeOffset + subPositionSize);

            for (auto output : nestedSimulate.getOutputs())
              sampledValues.push_back(output);
          }
        }

        rewriter.replaceAllUsesWith(sampleOp->getResults(), sampledValues);

        toErase.push_back(sampleOp);
        return success();
      };

      LogicalResult result = processImpulseBlock(
          NewF.getFunctionBody().front(), rewriter, carry, processSample);

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (failed(result)) {
        CI.emitError("Impulse: failed to walk sample ops");
        return failure();
      }

      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);
        SmallVector<Value> newRetVals;
        Value outTrace =
            scoreTailSize > 0
                ? appendScoreTail(rewriter, retOp.getLoc(), carry.currTrace,
                                  siteScores, declaredWidth)
                : carry.currTrace;
        newRetVals.push_back(outTrace);
        newRetVals.push_back(carry.weight);
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), CI.getInputs());

      rewriter.replaceOp(CI, newCI.getResults());

      return success();
    }
  };

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
    FailureOr<InferModel> resolveModel(MCMCOpT mcmcOp,
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
          mcmcOp.emitError("Impulse: calling `mcmc` on an empty function");
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

      InferModel model = readInferModel(mcmcOp);
      auto sizesFromSupports = [&]() {
        model.constrainedPositionSize = 0;
        model.positionSize = 0;
        for (const auto &info : model.supports) {
          model.constrainedPositionSize += info.constrainedSize;
          model.positionSize += info.unconstrainedSize;
        }
      };

      if (hasLogpdfFn) {
        auto initPosType =
            cast<RankedTensorType>(mcmcOp.getInitPosition().getType());
        // `init_position` is `1xN` or chain-major `Cx1xN`; N is per chain.
        model.positionSize =
            initPosType.getNumElements() / mcmcOp.getNumChains();
        model.constrainedPositionSize = model.positionSize;
      } else if constexpr (isRegionForm) {
        if (failed(verifySelectedSupports(mcmcOp)))
          return failure();
        model.supports = collectSupportInfoForRegion(mcmcOp);
        if (model.supports.empty()) {
          mcmcOp.emitError("Impulse: failed to collect per-site support info "
                           "from the infer_region sampler");
          return failure();
        }
        sizesFromSupports();
        if (model.positionSize <= 0)
          return failure();
      } else {
        for (Attribute address : model.selection) {
          if (cast<ArrayAttr>(address).size() > 1) {
            mcmcOp.emitError("Impulse: HMC and NUTS on a model take only "
                             "top-level addresses in `selection`, not ")
                << address;
            return failure();
          }
        }
        auto fn = cast<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));
        if (computePositionSizeForSelection(mcmcOp, fn, model.selection,
                                            symbolTable) <= 0)
          return failure();
        if (failed(verifySelectedSupports(mcmcOp, fn, model.selection)))
          return failure();

        model.supports = collectSupportInfoForSelection(
            mcmcOp, fn, model.selection, model.allAddresses, symbolTable,
            /*positionDeclarationOrder=*/false);
        sizesFromSupports();

        auto fnType = cast<FunctionType>(fn.getFunctionType());
        model.fnResultTypes.assign(fnType.getResults().begin(),
                                   fnType.getResults().end());
      }
      return model;
    }

    LogicalResult matchAndRewrite(MCMCOpT mcmcOp,
                                  PatternRewriter &rewriter) const override {
      FailureOr<InferModel> model = resolveModel(mcmcOp, rewriter);
      if (failed(model))
        return failure();

      MCMCChainEmitter emitter(rewriter, mcmcOp, std::move(*model), debugDump);
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
      Value stepSize = op.getInitStepSize();

      if (op.getInputs().empty()) {
        op.emitError("Impulse VI: requires at least rng_state input");
        return failure();
      }

      InferModel model = readInferModel(op);
      RegionVILayout regionLayout;

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
        if (!collectRegionVILayout(op, regionLayout)) {
          op.emitError("Impulse VI: failed to derive position layout from the "
                       "infer_region sampler");
          return failure();
        }
        model.supports = regionLayout.supports;
        model.positionSize = regionLayout.positionSize;
        if (model.positionSize <= 0)
          return failure();
      } else if (hasLogpdfFn) {
        model.positionSize =
            cast<RankedTensorType>(op.getInitPosition().getType())
                .getNumElements();
      } else {
        auto fn = cast<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
        model.positionSize = computePositionSizeForSelection(
            op, fn, model.selection, symbolTable);
        if (model.positionSize <= 0)
          return failure();
        if (failed(verifySelectedSupports(op, fn, model.selection)))
          return failure();

        model.supports = collectSupportInfoForSelection(
            op, fn, model.selection, model.allAddresses, symbolTable,
            /*positionDeclarationOrder=*/true);

        auto fnType = cast<FunctionType>(fn.getFunctionType());
        model.fnResultTypes.assign(fnType.getResults().begin(),
                                   fnType.getResults().end());
      }

      auto viConfig = op.getViConfig().value();
      auto floatOr = [](FloatAttr attr, double fallback) {
        return attr ? attr.getValueAsDouble() : fallback;
      };

      VI::VIContext viCtx(
          model.fn, model.fnInputs, model.fnResultTypes, model.originalTrace,
          model.allAddresses, stepSize, model.positionSize, op.getNumSamples(),
          model.supports, viConfig.getGuideKind(), viConfig.getOptimizerKind(),
          viConfig.getNumParticles(), floatOr(viConfig.getInitScale(), 0.1),
          floatOr(viConfig.getAdamB1(), 0.9),
          floatOr(viConfig.getAdamB2(), 0.999),
          floatOr(viConfig.getAdamEps(), 1.0e-8));
      viCtx.logpdfFn = model.logpdfFn;
      viCtx.autodiffAttrs = model.autodiffAttrs;
      viCtx.debugDump = debugDump;
      viCtx.clipNorm = floatOr(viConfig.getClipNorm(), viCtx.clipNorm);
      viCtx.initRadius = floatOr(viConfig.getInitRadius(), viCtx.initRadius);
      viCtx.momentumMass =
          floatOr(viConfig.getMomentumMass(), viCtx.momentumMass);
      viCtx.rmspropGamma =
          floatOr(viConfig.getRmspropGamma(), viCtx.rmspropGamma);
      viCtx.rmspropEps = floatOr(viConfig.getRmspropEps(), viCtx.rmspropEps);
      viCtx.stableUpdate = viConfig.getStableUpdate();
      viCtx.logpdfRegion = model.logpdfRegion;
      if constexpr (isRegionForm)
        viCtx.logpdfArgSlices = regionLayout.argSlices;

      return VI::lowerVI(rewriter, op, viCtx);
    }
  };

  template <typename OpT>
  struct LowerMHInferPattern : public mlir::OpRewritePattern<OpT> {
    static constexpr bool isRegionForm =
        std::is_same_v<OpT, impulse::InferRegionOp>;
    using mlir::OpRewritePattern<OpT>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpT op,
                                  PatternRewriter &rewriter) const override {
      if (!op.getMhConfig().has_value())
        return rewriter.notifyMatchFailure(op, "no mh_config; not an MH op");

      Location loc = op.getLoc();
      auto inputs = op.getInputs();
      if (inputs.empty()) {
        op.emitError("Impulse MH: requires at least the rng_state input");
        return failure();
      }
      if (op.getStateIn().size() != 1) {
        op.emitError("Impulse MH: expects a single `weight` state_in slot");
        return failure();
      }
      if constexpr (isRegionForm) {
        if (op.getSampling().empty() || op.getLogpdf().empty()) {
          op.emitError("Impulse MH: infer_region requires sampler and "
                       "constructed logpdf regions (run sicm first)");
          return failure();
        }
        unsigned numSites = 0;
        for (Operation &o : op.getSampling().front())
          numSites += isa<impulse::SampleRegionOp>(&o);
        if (op.getLogpdf().front().getTerminator()->getNumOperands() !=
            1 + numSites) {
          op.emitError("Impulse MH: infer_region logpdf region must yield the "
                       "joint density followed by one density per site");
          return failure();
        }
      }

      Value initTrace = op.getOriginalTrace();
      Value initWeight = op.getStateIn()[0];
      Value initRng = inputs[0];
      auto traceType = cast<RankedTensorType>(initTrace.getType());
      auto weightType = cast<RankedTensorType>(initWeight.getType());
      Type rngType = initRng.getType();
      auto i1TensorType = RankedTensorType::get({}, rewriter.getI1Type());

      ArrayAttr selection = op.getSelectionAttr();
      ArrayAttr allAddresses = op.getAllAddressesAttr();

      SymbolTableCollection symbolTable;
      FunctionOpInterface fn;
      SmallVector<Type> fnResultTypes;
      if constexpr (!isRegionForm) {
        fn = dyn_cast_or_null<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
        if (!fn || fn.getFunctionBody().empty()) {
          op.emitError("Impulse MH: `fn` must reference a non-empty funcOp");
          return failure();
        }
        for (Type t : fn.getResultTypes())
          fnResultTypes.push_back(t);
      }

      // A trace wider than the positions carries a score tail (delta weight).
      int64_t fullPositionSize = 0;
      RegionTraceLayout layout;
      if constexpr (isRegionForm) {
        FailureOr<RegionTraceLayout> regionLayout =
            getRegionTraceLayout(op, /*topLevelOnly=*/true);
        if (failed(regionLayout)) {
          op.emitError("Impulse MH: all_addresses entry without a top-level "
                       "site");
          return failure();
        }
        layout = std::move(*regionLayout);
        for (auto site : op.getSampling()
                             .front()
                             .template getOps<impulse::SampleRegionOp>())
          fullPositionSize += sampleRegionSize(site);
      } else {
        fullPositionSize =
            computePositionSizeForSelection(op, fn, allAddresses, symbolTable);
      }
      bool deltaWeight = traceType.getShape()[1] > fullPositionSize;

      // The region form fills this in during the proposal.
      ColumnWindows changedColumns;
      if constexpr (!isRegionForm)
        changedColumns =
            mhChangedColumnsFlat(op, fn, allAddresses, selection,
                                 traceType.getShape()[1], symbolTable);

      auto oneStep = [&](OpBuilder &b, Value trace, Value weight,
                         Value rng) -> MHStep {
        MHProposal proposal;
        if constexpr (isRegionForm)
          proposal =
              proposeRegionMove(b, loc, op, layout, trace, rng, deltaWeight,
                                /*withLogRatio=*/true, changedColumns);
        else
          proposal = proposeFlatMove(b, loc, op, fnResultTypes, trace, rng);
        MHDecision decision =
            emitMHDecision(b, loc, weight, proposal, deltaWeight);
        Value newTrace = writeBackMHWindows(b, loc, trace, proposal.trace,
                                            decision.accepted, changedColumns);
        Value newWeight = arith::SelectOp::create(
            b, loc, decision.accepted, decision.acceptWeight, weight);
        return {newTrace, newWeight, decision.rng, decision.accepted};
      };

      int64_t numSamples = op.getNumSamples();
      Value outTrace, outWeight, outRng, outAccepted;

      if (numSamples <= 1) {
        MHStep step = oneStep(rewriter, initTrace, initWeight, initRng);
        outTrace = step.trace;
        outWeight = step.weight;
        outRng = step.rng;
        outAccepted = step.accepted;
      } else {
        Value c0 = emitI64ScalarConstant(rewriter, loc, 0),
              cN = emitI64ScalarConstant(rewriter, loc, numSamples),
              c1 = emitI64ScalarConstant(rewriter, loc, 1);
        Value falseC = arith::ConstantOp::create(
            rewriter, loc, i1TensorType,
            DenseElementsAttr::get(i1TensorType, rewriter.getBoolAttr(false)));
        SmallVector<Type> loopTypes{traceType, weightType, rngType,
                                    i1TensorType};
        SmallVector<Value> initArgs{initTrace, initWeight, initRng, falseC};
        auto forOp = impulse::ForOp::create(rewriter, loc, loopTypes, c0, cN,
                                            c1, initArgs);
        Block *body = rewriter.createBlock(&forOp.getRegion());
        body->addArgument(c0.getType(), loc);
        for (Type t : loopTypes)
          body->addArgument(t, loc);
        rewriter.setInsertionPointToStart(body);
        MHStep step = oneStep(rewriter, body->getArgument(1),
                              body->getArgument(2), body->getArgument(3));
        impulse::YieldOp::create(
            rewriter, loc,
            ValueRange{step.trace, step.weight, step.rng, step.accepted});
        rewriter.setInsertionPointAfter(forOp);
        outTrace = forOp.getResult(0);
        outWeight = forOp.getResult(1);
        outRng = forOp.getResult(2);
        outAccepted = forOp.getResult(3);
      }

      // MH reports no per-sample log density; `log_densities` is zero.
      Value outLogDensities = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(op.getLogDensities().getType()));
      rewriter.replaceOp(
          op, {outTrace, outAccepted, outLogDensities, outRng, outWeight});
      return success();
    }
  };

  /// Enumerative Gibbs over scalar BERNOULLI sites: new = u < logistic(s1 -
  /// s0). The batched sweep is sound only when each per-site yield depends on
  /// at most one selected site and no site's parameters depend on another.
  struct LowerGibbsRegionPattern
      : public mlir::OpRewritePattern<impulse::InferRegionOp> {
    using mlir::OpRewritePattern<impulse::InferRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::InferRegionOp op,
                                  PatternRewriter &rewriter) const override {
      if (!op.getGibbsConfig().has_value())
        return rewriter.notifyMatchFailure(op, "no gibbs_config");
      Location loc = op.getLoc();
      if (op.getSampling().empty() || op.getLogpdf().empty()) {
        op.emitError("Impulse Gibbs: infer_region requires sampler and "
                     "constructed logpdf regions (run sicm first)");
        return failure();
      }
      if (op.getStateIn().size() != 1 || op.getInputs().empty() ||
          !op.getOriginalTrace()) {
        op.emitError("Impulse Gibbs: expects the rng input, a trace and a "
                     "single `weight` state_in slot");
        return failure();
      }
      Block &samplerEntry = op.getSampling().front();
      Block &lpBlock = op.getLogpdf().front();
      Operation *lpTerm = lpBlock.getTerminator();
      unsigned numYields = lpTerm->getNumOperands() - 1;
      ArrayAttr selection = op.getSelectionAttr();

      FailureOr<RegionTraceLayout> layout =
          getRegionTraceLayout(op, /*topLevelOnly=*/false);
      if (failed(layout)) {
        op.emitError("Impulse Gibbs: all_addresses entry without a site");
        return failure();
      }
      DenseMap<Attribute, impulse::SampleRegionOp> &siteByAddress =
          layout->siteByAddress;
      int64_t positionSize = layout->positionSize;
      Value initTrace = op.getOriginalTrace();
      Value initWeight = op.getStateIn()[0];
      Value initRng = op.getInputs()[0];
      auto traceType = cast<RankedTensorType>(initTrace.getType());
      auto weightType = cast<RankedTensorType>(initWeight.getType());
      Type rngType = initRng.getType();
      int64_t width = traceType.getShape()[1];
      bool scoreTail = width > positionSize;
      if (scoreTail && width != positionSize + (int64_t)numYields) {
        op.emitError("Impulse Gibbs: score-tail width does not match the "
                     "logpdf region's per-site yields");
        return failure();
      }
      if (lpBlock.getNumArguments() != selection.size()) {
        op.emitError("Impulse Gibbs: one position argument per selected "
                     "site expected");
        return failure();
      }
      SmallVector<int64_t> selectedCols;
      for (auto [k, addr] : llvm::enumerate(selection)) {
        impulse::SampleRegionOp site = siteByAddress.lookup(addr);
        impulse::DistAttr dist = site ? site.getDistAttr() : nullptr;
        if (!site || site->getBlock() != &samplerEntry || !dist ||
            dist.getBase() != impulse::DistBaseKind::BERNOULLI ||
            (dist.getTransforms() && !dist.getTransforms().empty()) ||
            site.getNumResults() != 2 ||
            cast<RankedTensorType>(lpBlock.getArgument(k).getType())
                    .getNumElements() != 1) {
          op.emitError("Impulse Gibbs: every selected site must be a "
                       "top-level scalar BERNOULLI site");
          return failure();
        }
        selectedCols.push_back(layout->columnOf.lookup(addr));
      }
      // (trace col, constraint offset, size) of constrained branch-local sites.
      SmallVector<std::tuple<int64_t, int64_t, int64_t>> observedBranchCols;
      if (Value constraint = op.getConstraint()) {
        int64_t off = 0;
        for (Attribute addr : op.getConstrainedAddressesAttr()) {
          impulse::SampleRegionOp site = siteByAddress.lookup(addr);
          if (!site) {
            op.emitError("Impulse Gibbs: constrained address without a site");
            return failure();
          }
          if (site->getBlock() != &samplerEntry)
            observedBranchCols.push_back(
                {layout->columnOf.lookup(addr), off, sampleRegionSize(site)});
          off += sampleRegionSize(site);
        }
      }
      SmallVector<SmallVector<bool>> dirtyOf;
      for (unsigned k = 0; k < selection.size(); ++k)
        dirtyOf.push_back(
            taintedSiteYields(lpBlock, ValueRange{lpBlock.getArgument(k)}));

      // siteOfYield[j]: the selected site yield j depends on, or -1 for none.
      unsigned numSites = selection.size();
      SmallVector<int> siteOfYield(numYields, -1);
      bool batched = numSites >= 2;
      if (batched) {
        SiteDependence lpDep(numSites);
        for (unsigned k = 0; k < numSites; ++k)
          lpDep.seed(lpBlock.getArgument(k), k);
        lpDep.run(lpBlock);
        for (unsigned j = 0; j < numYields && batched; ++j) {
          llvm::BitVector s = lpDep.get(lpTerm->getOperand(1 + j));
          if (s.count() > 1)
            batched = false;
          else if (s.any())
            siteOfYield[j] = s.find_first();
        }
      }
      if (batched) {
        SiteDependence samplerDep(numSites);
        SmallVector<impulse::SampleRegionOp> selectedSites;
        for (auto [k, addr] : llvm::enumerate(selection)) {
          impulse::SampleRegionOp site = siteByAddress.lookup(addr);
          selectedSites.push_back(site);
          samplerDep.seed(site.getResult(1), k);
        }
        samplerDep.run(samplerEntry);
        for (auto [k, site] : llvm::enumerate(selectedSites))
          for (Value param : llvm::drop_begin(site->getOperands()))
            if (samplerDep.get(param).any())
              batched = false;
      }

      auto writeCols = [&](OpBuilder &b, Value trace, int64_t col,
                           Value value) -> Value {
        auto vt = cast<RankedTensorType>(value.getType());
        auto flatType = RankedTensorType::get({1, vt.getNumElements()},
                                              traceType.getElementType());
        Value flat = impulse::ReshapeOp::create(b, loc, flatType, value);
        return updateColumns(b, loc, trace, flat, col);
      };

      // Readers of the trace take constrained branch-local sites from it, so
      // it holds their observed values.
      auto writeObserved = [&](OpBuilder &b, Value trace) -> Value {
        Value constraint = op.getConstraint();
        for (auto [col, off, n] : observedBranchCols) {
          Value obs = impulse::SliceOp::create(
              b, loc, RankedTensorType::get({1, n}, traceType.getElementType()),
              constraint, b.getDenseI64ArrayAttr({0, off}),
              b.getDenseI64ArrayAttr({1, off + n}),
              b.getDenseI64ArrayAttr({1, 1}));
          trace = writeCols(b, trace, col, obs);
        }
        return trace;
      };

      auto batchedSweep = [&](OpBuilder &b, Value trace, Value weight,
                              Value rng) -> std::array<Value, 3> {
        int64_t n = numSites;
        Type elemType = traceType.getElementType();
        auto scalarType = RankedTensorType::get({}, elemType);
        auto vecType = RankedTensorType::get({n}, elemType);
        auto rowType = RankedTensorType::get({1, n}, elemType);
        auto colType = RankedTensorType::get({1, 1}, elemType);
        int64_t stride = n > 1 ? selectedCols[1] - selectedCols[0] : 1;
        bool evenlySpaced = stride >= 1;
        for (int64_t k = 1; k < n && evenlySpaced; ++k)
          evenlySpaced = selectedCols[k] - selectedCols[k - 1] == stride;

        Value curVec;
        if (evenlySpaced) {
          Value row = impulse::SliceOp::create(
              b, loc, rowType, trace,
              b.getDenseI64ArrayAttr({0, selectedCols[0]}),
              b.getDenseI64ArrayAttr({1, selectedCols[n - 1] + 1}),
              b.getDenseI64ArrayAttr({1, stride}));
          curVec = impulse::ReshapeOp::create(b, loc, vecType, row);
        } else {
          SmallVector<Value> cur;
          for (int64_t col : selectedCols)
            cur.push_back(scalarAt(b, loc, trace, {0, col}));
          curVec = packScalars(b, loc, vecType, cur);
        }

        // All sites at v at once: each yield sees at most one selected site.
        SmallVector<Value> yieldsAt[2];
        for (int v = 0; v < 2; ++v) {
          SmallVector<Value> positions;
          for (BlockArgument arg : lpBlock.getArguments())
            positions.push_back(emitSplatConstant(
                b, loc, cast<RankedTensorType>(arg.getType()), (double)v));
          llvm::append_range(yieldsAt[v], llvm::drop_begin(cloneLogpdfBody(
                                              b, lpBlock, positions)));
        }
        SmallVector<Value> logOdds(n);
        for (unsigned j = 0; j < numYields; ++j) {
          int k = siteOfYield[j];
          if (k < 0)
            continue;
          Value d =
              arith::SubFOp::create(b, loc, yieldsAt[1][j], yieldsAt[0][j]);
          logOdds[k] =
              logOdds[k]
                  ? arith::AddFOp::create(b, loc, logOdds[k], d).getResult()
                  : d;
        }
        for (Value &l : logOdds)
          if (!l)
            l = emitSplatConstant(b, loc, scalarType, 0.0);
        Value lVec = packScalars(b, loc, vecType, logOdds);

        Value p1 = impulse::LogisticOp::create(b, loc, vecType, lVec);
        Value uniform = emitUniformDraw(b, loc, rng, vecType);
        Value take1 = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                            uniform, p1);
        Value curIs1 =
            arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, curVec,
                                  emitSplatConstant(b, loc, vecType, 0.5));
        Value zeros = emitSplatConstant(b, loc, vecType, 0.0);
        Value newVals = arith::SelectOp::create(
            b, loc, take1, emitSplatConstant(b, loc, vecType, 1.0), zeros);

        // +L_k on a 0 -> 1 flip, -L_k on 1 -> 0, else 0.
        Value negL = arith::NegFOp::create(b, loc, lVec);
        Value toOne = arith::SelectOp::create(b, loc, curIs1, zeros, lVec);
        Value toZero = arith::SelectOp::create(b, loc, curIs1, negL, zeros);
        Value delta = arith::SelectOp::create(b, loc, take1, toOne, toZero);
        for (int64_t k = 0; k < n; ++k)
          weight = arith::AddFOp::create(b, loc, weight,
                                         takeLeading(b, loc, delta, k));

        Value newRow = impulse::ReshapeOp::create(b, loc, rowType, newVals);
        if (evenlySpaced && stride == 1) {
          trace = updateColumns(b, loc, trace, newRow, selectedCols[0]);
        } else {
          for (auto [k, col] : llvm::enumerate(selectedCols)) {
            Value colSlice = impulse::SliceOp::create(
                b, loc, colType, newRow,
                b.getDenseI64ArrayAttr({0, (int64_t)k}),
                b.getDenseI64ArrayAttr({1, (int64_t)k + 1}),
                b.getDenseI64ArrayAttr({1, 1}));
            trace = updateColumns(b, loc, trace, colSlice, col);
          }
        }
        if (scoreTail)
          for (unsigned j = 0; j < numYields; ++j)
            if (siteOfYield[j] >= 0)
              trace = writeCols(b, trace, positionSize + j,
                                arith::SelectOp::create(
                                    b, loc,
                                    takeLeading(b, loc, take1, siteOfYield[j]),
                                    yieldsAt[1][j], yieldsAt[0][j]));
        return {writeObserved(b, trace), weight, rng};
      };

      auto oneSweep = [&](OpBuilder &b, Value trace, Value weight,
                          Value rng) -> std::array<Value, 3> {
        if (batched)
          return batchedSweep(b, trace, weight, rng);
        SmallVector<Value> cur;
        for (auto [k, col] : llvm::enumerate(selectedCols))
          cur.push_back(sliceTraceValue(
              b, loc, trace, col,
              cast<RankedTensorType>(lpBlock.getArgument(k).getType())));
        auto scalarType = RankedTensorType::get({}, traceType.getElementType());
        for (unsigned k = 0; k < selectedCols.size(); ++k) {
          auto argType =
              cast<RankedTensorType>(lpBlock.getArgument(k).getType());
          SmallVector<Value> yieldsAt[2];
          Value scoreAt[2];
          for (int v = 0; v < 2; ++v) {
            SmallVector<Value> positions(cur);
            positions[k] = emitSplatConstant(b, loc, argType, (double)v);
            SmallVector<Value> yields = cloneLogpdfBody(b, lpBlock, positions);
            for (unsigned j = 0; j < numYields; ++j) {
              Value y = yields[1 + j];
              yieldsAt[v].push_back(y);
              if (!dirtyOf[k][j])
                continue;
              scoreAt[v] =
                  scoreAt[v]
                      ? arith::AddFOp::create(b, loc, scoreAt[v], y).getResult()
                      : y;
            }
            if (!scoreAt[v])
              scoreAt[v] = emitSplatConstant(b, loc, scalarType, 0.0);
          }
          Value diff = arith::SubFOp::create(b, loc, scoreAt[1], scoreAt[0]);
          Value p1 = impulse::LogisticOp::create(b, loc, diff.getType(), diff);
          Value uniform = emitUniformDraw(b, loc, rng, weightType);
          Value take1 = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                              uniform, p1);
          Value curScalar =
              impulse::ReshapeOp::create(b, loc, scalarType, cur[k]);
          Value curIs1 = arith::CmpFOp::create(
              b, loc, arith::CmpFPredicate::OGT, curScalar,
              emitSplatConstant(b, loc, scalarType, 0.5));
          Value sNew =
              arith::SelectOp::create(b, loc, take1, scoreAt[1], scoreAt[0]);
          Value sCur =
              arith::SelectOp::create(b, loc, curIs1, scoreAt[1], scoreAt[0]);
          weight = arith::AddFOp::create(
              b, loc, weight, arith::SubFOp::create(b, loc, sNew, sCur));
          Value oneArg = emitSplatConstant(b, loc, argType, 1.0);
          Value zeroArg = emitSplatConstant(b, loc, argType, 0.0);
          Value newVal =
              arith::SelectOp::create(b, loc, take1, oneArg, zeroArg);
          cur[k] = newVal;
          trace = writeCols(b, trace, selectedCols[k], newVal);
          if (scoreTail)
            for (unsigned j = 0; j < numYields; ++j)
              if (dirtyOf[k][j])
                trace = writeCols(b, trace, positionSize + j,
                                  arith::SelectOp::create(b, loc, take1,
                                                          yieldsAt[1][j],
                                                          yieldsAt[0][j]));
        }
        return {writeObserved(b, trace), weight, rng};
      };

      int64_t numSamples = op.getNumSamples();
      Value outTrace, outWeight, outRng;
      if (numSamples <= 1) {
        auto r = oneSweep(rewriter, initTrace, initWeight, initRng);
        outTrace = r[0];
        outWeight = r[1];
        outRng = r[2];
      } else {
        Value c0 = emitI64ScalarConstant(rewriter, loc, 0),
              cN = emitI64ScalarConstant(rewriter, loc, numSamples),
              c1 = emitI64ScalarConstant(rewriter, loc, 1);
        SmallVector<Type> loopTypes{traceType, weightType, rngType};
        SmallVector<Value> initArgs{initTrace, initWeight, initRng};
        auto forOp = impulse::ForOp::create(rewriter, loc, loopTypes, c0, cN,
                                            c1, initArgs);
        Block *body = rewriter.createBlock(&forOp.getRegion());
        body->addArgument(c0.getType(), loc);
        for (Type t : loopTypes)
          body->addArgument(t, loc);
        rewriter.setInsertionPointToStart(body);
        auto r = oneSweep(rewriter, body->getArgument(1), body->getArgument(2),
                          body->getArgument(3));
        impulse::YieldOp::create(rewriter, loc, ValueRange{r[0], r[1], r[2]});
        rewriter.setInsertionPointAfter(forOp);
        outTrace = forOp.getResult(0);
        outWeight = forOp.getResult(1);
        outRng = forOp.getResult(2);
      }

      // Gibbs always accepts; `log_densities` is a zero placeholder.
      auto diagType = cast<ShapedType>(op.getDiagnostics().getType());
      Value diagnostics = arith::ConstantOp::create(
          rewriter, loc, diagType,
          DenseElementsAttr::get(diagType, rewriter.getBoolAttr(true)));
      Value logDensities = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(op.getLogDensities().getType()));
      rewriter.replaceOp(
          op, {outTrace, diagnostics, logDensities, outRng, outWeight});
      return success();
    }
  };

  struct RejectFlatGibbsPattern
      : public mlir::OpRewritePattern<impulse::InferOp> {
    using mlir::OpRewritePattern<impulse::InferOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(impulse::InferOp op,
                                  PatternRewriter &rewriter) const override {
      if (!op.getGibbsConfig().has_value())
        return failure();
      op.emitError("Impulse Gibbs: requires the region pipeline "
                   "(inline-infer-regions, sicm); this model was not admitted "
                   "to region form");
      return failure();
    }
  };

  struct LowerMHPattern : public mlir::OpRewritePattern<impulse::MHOp> {
    using mlir::OpRewritePattern<impulse::MHOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::MHOp mhOp,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mhOp, mhOp.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        mhOp.emitError(
            "Impulse: calling `mh` on an empty function; if this is a "
            "distribution function, its sample op should have a logpdf "
            "attribute to avoid recursive `mh` calls which is intended for "
            "generative functions");
        return failure();
      }

      auto loc = mhOp.getLoc();

      Value oldTrace = mhOp.getOriginalTrace();
      Value oldWeight = mhOp.getOriginalWeight();
      SmallVector<Value> inputs(mhOp.getInputs().begin(),
                                mhOp.getInputs().end());
      auto allAddresses = mhOp.getAllAddressesAttr();

      auto traceType = oldTrace.getType();

      auto nameAttr = mhOp.getNameAttr();
      if (!nameAttr)
        nameAttr = rewriter.getStringAttr("");

      auto selection = mhOp.getSelectionAttr();

      // On a score-tail trace, regenerate returns the dirty-term weight delta.
      int64_t fullPositionSize =
          computePositionSizeForSelection(mhOp, fn, allAddresses, symbolTable);
      bool delta =
          cast<RankedTensorType>(traceType).getShape()[1] > fullPositionSize;

      SmallVector<Type> regenResultTypes;
      regenResultTypes.push_back(traceType);
      regenResultTypes.push_back(
          RankedTensorType::get({2}, rewriter.getF64Type()));
      for (auto t : fn.getResultTypes())
        regenResultTypes.push_back(t);

      auto regenerateOp = impulse::RegenerateOp::create(
          rewriter, loc, regenResultTypes, mhOp.getFnAttr(), inputs, oldTrace,
          mhOp.getConstraint(), allAddresses, selection,
          mhOp.getConstrainedAddressesAttr(), nameAttr);

      regenerateOp->setAttr(kMHHastingsAttr, rewriter.getUnitAttr());
      MHProposal proposal;
      proposal.trace = regenerateOp.getNewTrace();
      std::tie(proposal.weight, proposal.corr) =
          unpackHastingsWeight(rewriter, loc, regenerateOp.getWeight());
      proposal.rng = regenerateOp.getOutputs()[0];

      MHDecision decision =
          emitMHDecision(rewriter, loc, oldWeight, proposal, delta);
      Value newTrace = writeBackMHWindows(
          rewriter, loc, oldTrace, proposal.trace, decision.accepted,
          mhChangedColumnsFlat(mhOp, fn, allAddresses, selection,
                               cast<RankedTensorType>(traceType).getShape()[1],
                               symbolTable));
      Value newWeight = arith::SelectOp::create(
          rewriter, loc, decision.accepted, decision.acceptWeight, oldWeight);
      rewriter.replaceOp(
          mhOp, {newTrace, newWeight, decision.accepted, decision.rng});
      return success();
    }
  };

  struct LowerGeneratePattern
      : public mlir::OpRewritePattern<impulse::GenerateOp> {
    using mlir::OpRewritePattern<impulse::GenerateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::GenerateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "Impulse: calling `generate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "`logpdf_fn` attribute to avoid recursive `generate` calls "
            "which is intended for generative functions");
        return failure();
      }

      ArrayAttr allAddresses = CI.getAllAddressesAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, allAddresses, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("Impulse: failed to compute position size for generate");
        return failure();
      }

      // Operand width, not the packed size: extra columns are simply ignored.
      int64_t constraintSize =
          cast<RankedTensorType>(CI.getConstraint().getType()).getShape()[1];

      int64_t declaredWidth =
          cast<RankedTensorType>(CI.getTrace().getType()).getShape()[1];
      int64_t scoreTailSize = declaredWidth - positionSize;
      SmallVector<Value> siteScores;
      std::unique_ptr<ImpulseUtils> putils(ImpulseUtils::CreateFromClone(
          fn, ImpulseMode::Generate, declaredWidth, constraintSize));
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();

      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      Value zeroWeight =
          emitSplatConstant(entryBuilder, initLoc, scalarType, 0.0);

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      Value zeroTrace =
          emitSplatConstant(entryBuilder, initLoc, traceType, 0.0);
      Value constraint = NewF.getArgument(0);

      SmallVector<Operation *> toErase;
      BranchCarry carry{zeroTrace, zeroWeight, /*offset=*/0};
      auto processSample = [&](impulse::SampleOp sampleOp,
                               BranchCarry &branch) -> LogicalResult {
        Value &currTrace = branch.currTrace;
        Value &weightAccumulator = branch.weight;
        int64_t &currentTraceOffset = branch.offset;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        if (sampleOp.getDistAttr() || sampleOp.getLogpdfFnAttr()) {
          if (failed(verifyTopLevelAddress(
                  sampleOp, CI.getConstrainedAddressesAttr(), "constrained")))
            return failure();
          bool isConstrained =
              addressInList(sampleOp, CI.getConstrainedAddressesAttr());
          if (isConstrained) {
            int64_t constrainedOffset = computeOffsetForSampleInSelection(
                CI, fn, CI.getConstrainedAddressesAttr(),
                sampleOp.getSymbolAttr(), symbolTable);
            FailureOr<SmallVector<Value>> read = readSiteValues(
                rewriter, sampleOp, constraint, constrainedOffset);
            if (failed(read))
              return failure();
            sampledValues = std::move(*read);
          } else {
            if (impulse::DistAttr dist = sampleOp.getDistAttr();
                dist && !impulse::distSamplerEmittable(dist)) {
              sampleOp.emitError("Impulse: distribution sampler is not "
                                 "emittable; the site must be constrained");
              return failure();
            }
            sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          }
          ArrayRef<Value> values = ArrayRef<Value>(sampledValues).drop_front();
          if (failed(verifySiteLogpdf(symbolTable, sampleOp, values.size(),
                                      isConstrained
                                          ? "logpdf call for constrained sample"
                                          : "logpdf call")))
            return failure();
          Value logpdf =
              emitSiteLogpdf(rewriter, symbolTable, sampleOp, values);
          weightAccumulator = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                    weightAccumulator, logpdf);
          if (scoreTailSize > 0)
            siteScores.push_back(logpdf);
          if (addressInList(sampleOp, allAddresses) &&
              failed(appendSiteValues(rewriter, sampleOp, values, currTrace,
                                      currentTraceOffset)))
            return failure();
        } else {
          if (scoreTailSize > 0) {
            sampleOp.emitError("Impulse: score-tail traces do not support "
                               "nested generative sites");
            return failure();
          }
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "Impulse: generative function body is empty; "
                "if this is a distribution, add a `logpdf_fn` attribute");
            return failure();
          }

          ArrayAttr subAllAddresses = buildSubSelection(
              rewriter, allAddresses, sampleOp.getSymbolAttr());
          ArrayAttr subConstrainedAddrs =
              buildSubSelection(rewriter, CI.getConstrainedAddressesAttr(),
                                sampleOp.getSymbolAttr());

          if (subAllAddresses.empty()) {
            sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subAllAddresses, symbolTable);
            int64_t subConstraintSize = computePositionSizeForSelection(
                sampleOp, genFn, subConstrainedAddrs, symbolTable);
            if (subPositionSize <= 0 || subConstraintSize < 0) {
              sampleOp.emitError("Impulse: failed to compute sub-position or "
                                 "sub-constraint size");
              return failure();
            }

            Value subConstraint;
            auto subConstraintType = RankedTensorType::get(
                {1, subConstraintSize}, rewriter.getF64Type());

            if (subConstraintSize > 0) {
              int64_t subConstraintOffset = computeOffsetForSampleInSelection(
                  sampleOp, fn, CI.getConstrainedAddressesAttr(),
                  sampleOp.getSymbolAttr(), symbolTable);

              subConstraint = impulse::SliceOp::create(
                  rewriter, sampleOp.getLoc(), subConstraintType, constraint,
                  rewriter.getDenseI64ArrayAttr({0, subConstraintOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, subConstraintOffset + subConstraintSize}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
            } else {
              subConstraint = emitSplatConstant(rewriter, sampleOp.getLoc(),
                                                subConstraintType, 0.0);
            }

            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> genResultTypes;
            genResultTypes.push_back(subTraceType);
            genResultTypes.push_back(scalarTy);
            for (auto t : genFn.getResultTypes())
              genResultTypes.push_back(t);

            auto nestedGenerate = impulse::GenerateOp::create(
                rewriter, sampleOp.getLoc(), genResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subConstraint,
                subAllAddresses, subConstrainedAddrs);

            Value subTrace = nestedGenerate.getTrace();
            Value subWeight = nestedGenerate.getWeight();

            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);

            int64_t mergeOffset = computeOffsetForSampleInSelection(
                sampleOp, fn, allAddresses, sampleOp.getSymbolAttr(),
                symbolTable);

            currTrace = updateColumns(rewriter, sampleOp.getLoc(), currTrace,
                                      subTrace, mergeOffset);
            currentTraceOffset =
                std::max(currentTraceOffset, mergeOffset + subPositionSize);

            for (auto output : nestedGenerate.getOutputs())
              sampledValues.push_back(output);
          }
        }

        rewriter.replaceAllUsesWith(sampleOp->getResults(), sampledValues);
        toErase.push_back(sampleOp);
        return success();
      };

      LogicalResult result = processImpulseBlock(
          NewF.getFunctionBody().front(), rewriter, carry, processSample);

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (failed(result)) {
        CI.emitError("Impulse: failed to walk sample ops");
        return failure();
      }

      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

        SmallVector<Value> newRetVals;
        Value outTrace =
            scoreTailSize > 0
                ? appendScoreTail(rewriter, retOp.getLoc(), carry.currTrace,
                                  siteScores, declaredWidth)
                : carry.currTrace;
        newRetVals.push_back(outTrace);
        newRetVals.push_back(carry.weight);
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getConstraint());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      return success();
    }
  };

  struct LowerRegeneratePattern
      : public mlir::OpRewritePattern<impulse::RegenerateOp> {
    using mlir::OpRewritePattern<impulse::RegenerateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::RegenerateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "Impulse: calling `regenerate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "`logpdf_fn` attribute to avoid recursive `regenerate` calls "
            "which is intended for generative functions");
        return failure();
      }

      ArrayAttr allAddresses = CI.getAllAddressesAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, allAddresses, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("Impulse: failed to compute position size for regenerate");
        return failure();
      }

      ArrayAttr constrainedAddresses = CI.getConstrainedAddressesAttr();
      bool hasConstraint = static_cast<bool>(CI.getConstraint());
      // Operand width, not the packed size, as in generate.
      int64_t constraintSize =
          hasConstraint ? cast<RankedTensorType>(CI.getConstraint().getType())
                              .getShape()[1]
                        : -1;

      int64_t declaredWidth =
          cast<RankedTensorType>(CI.getNewTrace().getType()).getShape()[1];
      int64_t scoreTailSize = declaredWidth - positionSize;
      SmallVector<Value> siteScores;
      int64_t siteOrdinal = 0;
      std::unique_ptr<ImpulseUtils> putils(ImpulseUtils::CreateFromClone(
          fn, ImpulseMode::Regenerate, declaredWidth, constraintSize));
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();

      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      Value zeroWeight =
          emitSplatConstant(entryBuilder, initLoc, scalarType, 0.0);
      Value initWeight = zeroWeight;

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      Value zeroTrace =
          emitSplatConstant(entryBuilder, initLoc, traceType, 0.0);
      Value initTrace = zeroTrace;

      Value prevTrace = NewF.getArgument(0);
      Value constraint = hasConstraint ? NewF.getArgument(1) : Value();

      // On a score-tail trace the weight is the sum over dirty terms
      // (regenerated or dependent on one) of new minus old log density; clean
      // terms cancel in the MH ratio. Taint is computed up front; fresh draws
      // are added as they are created.
      bool deltaMode = scoreTailSize > 0;
      DenseSet<Value> tainted;
      if (deltaMode) {
        if (!deltaWeightFeasible(NewF)) {
          CI.emitError("Impulse: regenerate over a score-tail trace requires "
                       "a straight-line body of top-level distribution sites");
          return failure();
        }
        tainted = computeRegenerateTaint(NewF, CI.getSelectionAttr());
        // Non-regenerated values are unchanged: start from the previous trace.
        {
          auto valueSliceType = RankedTensorType::get(
              {1, positionSize}, entryBuilder.getF64Type());
          initTrace = impulse::SliceOp::create(
                          entryBuilder, initLoc, valueSliceType, prevTrace,
                          entryBuilder.getDenseI64ArrayAttr({0, 0}),
                          entryBuilder.getDenseI64ArrayAttr({1, positionSize}),
                          entryBuilder.getDenseI64ArrayAttr({1, 1}))
                          .getResult();
        }
      }

      ArrayAttr selection = CI.getSelectionAttr();
      bool hastings = CI->hasAttr(kMHHastingsAttr);
      ArrayAttr siteScoresAttr =
          CI->getAttrOfType<ArrayAttr>(kMHSiteScoresAttr);
      bool scoring = static_cast<bool>(siteScoresAttr);
      // Site symbol -> column of the scores vector or of `oldInfo`.
      DenseMap<Attribute, int64_t> siteIndex;
      Value oldInfo;
      DenseSet<Value> hastingsTaint;
      auto enclosingIfs = [&](Operation *op) {
        SmallVector<impulse::IfOp> ifs;
        for (Operation *p = op->getParentOp(); p && p != NewF.getOperation();
             p = p->getParentOp())
          if (auto ifOp = dyn_cast<impulse::IfOp>(p))
            ifs.push_back(ifOp);
        return ifs;
      };
      auto inFlippableBranch = [&](Operation *op) {
        return llvm::any_of(enclosingIfs(op), [&](impulse::IfOp ifOp) {
          return hastingsTaint.contains(ifOp.getPredicate());
        });
      };
      if (scoring) {
        for (auto [k, addr] : llvm::enumerate(siteScoresAttr))
          siteIndex[cast<ArrayAttr>(addr)[0]] = k;
        auto scoresType = RankedTensorType::get(
            {2, (int64_t)siteScoresAttr.size()}, entryBuilder.getF64Type());
        initWeight = emitSplatConstant(entryBuilder, initLoc, scoresType, 0.0);
      } else if (hastings && !deltaMode) {
        // Sites whose old density cannot be evaluated inline at the new state.
        hastingsTaint = computeRegenerateTaint(NewF, selection);
        SmallVector<Attribute> oldInfoAddrs;
        NewF.getFunctionBody().walk([&](impulse::SampleOp s) {
          if (!s.getDistAttr() && !s.getLogpdfFnAttr())
            return;
          bool needsOld;
          if (addressInList(s, selection))
            needsOld =
                !enclosingIfs(s).empty() ||
                llvm::any_of(llvm::drop_begin(s.getInputs()), [&](Value v) {
                  return hastingsTaint.contains(v);
                });
          else
            needsOld = !(constraint && constrainedAddresses &&
                         addressInList(s, constrainedAddresses)) &&
                       inFlippableBranch(s);
          if (!needsOld)
            return;
          siteIndex[s.getSymbolAttr()] = oldInfoAddrs.size();
          oldInfoAddrs.push_back(rewriter.getArrayAttr({s.getSymbolAttr()}));
        });
        if (!oldInfoAddrs.empty()) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&NewF.getFunctionBody().front());
          auto scoresType = RankedTensorType::get(
              {2, (int64_t)oldInfoAddrs.size()}, rewriter.getF64Type());
          SmallVector<Type> scoreTypes{prevTrace.getType(), scoresType};
          scoreTypes.append(fn.getResultTypes().begin(),
                            fn.getResultTypes().end());
          SmallVector<Value> scoreInputs(
              NewF.getArguments().drop_front(hasConstraint ? 2 : 1));
          auto scoreOp = impulse::RegenerateOp::create(
              rewriter, CI.getLoc(), scoreTypes, CI.getFnAttr(), scoreInputs,
              prevTrace, constraint, allAddresses, rewriter.getArrayAttr({}),
              hasConstraint ? constrainedAddresses : ArrayAttr(),
              rewriter.getStringAttr(""));
          scoreOp->setAttr(kMHSiteScoresAttr,
                           rewriter.getArrayAttr(oldInfoAddrs));
          oldInfo = scoreOp.getWeight();
        }
      }
      // In MH, constraint entries outside branches may be latent init values.
      auto readsConstraint = [&](impulse::SampleOp s) {
        if (!constraint || !constrainedAddresses)
          return false;
        return !(hastings || scoring) || !enclosingIfs(s).empty();
      };
      // `oldInfo` column of a flippable latent (kept if live, else redrawn).
      auto flipLatentIndex = [&](impulse::SampleOp s, bool isSelected) {
        if (!hastings || isSelected)
          return (int64_t)-1;
        auto it = siteIndex.find(s.getSymbolAttr());
        return it == siteIndex.end() ? (int64_t)-1 : it->second;
      };
      auto oldLiveAt = [&](Location loc, int64_t k) -> Value {
        Value flag = scalarAt(rewriter, loc, oldInfo, {0, k});
        Value half = emitSplatConstant(rewriter, loc, scalarType, 0.5);
        return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                     flag, half);
      };
      auto recordSiteScore = [&](Location loc, Value acc, int64_t k,
                                 Value score) -> Value {
        auto t11 = RankedTensorType::get({1, 1}, rewriter.getF64Type());
        Value one = emitSplatConstant(rewriter, loc, t11, 1.0);
        Value scoreRow = impulse::ReshapeOp::create(rewriter, loc, t11, score);
        acc = impulse::DynamicUpdateSliceOp::create(
            rewriter, loc, acc.getType(), acc, one,
            ValueRange{emitI64ScalarConstant(rewriter, loc, 0),
                       emitI64ScalarConstant(rewriter, loc, k)});
        return impulse::DynamicUpdateSliceOp::create(
            rewriter, loc, acc.getType(), acc, scoreRow,
            ValueRange{emitI64ScalarConstant(rewriter, loc, 1),
                       emitI64ScalarConstant(rewriter, loc, k)});
      };

      SmallVector<Operation *> toErase;
      BranchCarry carry{initTrace, initWeight, /*offset=*/0,
                        hastings ? Value(zeroWeight) : Value()};
      auto processSample = [&](impulse::SampleOp sampleOp,
                               BranchCarry &branch) -> LogicalResult {
        Value &currTrace = branch.currTrace;
        Value &weightAccumulator = branch.weight;
        int64_t &currentTraceOffset = branch.offset;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        if (sampleOp.getDistAttr() || sampleOp.getLogpdfFnAttr()) {
          Location loc = sampleOp.getLoc();
          if (failed(verifyTopLevelAddress(sampleOp, selection, "selected")))
            return failure();
          bool isSelected = addressInList(sampleOp, selection);
          int64_t sampleOffset = computeOffsetForSampleInSelection(
              CI, fn, allAddresses, sampleOp.getSymbolAttr(), symbolTable);

          int64_t flipIdx = flipLatentIndex(sampleOp, isSelected);
          Value oldLive;
          if (isSelected) {
            sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          } else {
            bool isConstrained = false;
            if (readsConstraint(sampleOp)) {
              if (failed(verifyTopLevelAddress(sampleOp, constrainedAddresses,
                                               "constrained")))
                return failure();
              isConstrained = addressInList(sampleOp, constrainedAddresses);
            }
            int64_t readOffset =
                isConstrained ? computeOffsetForSampleInSelection(
                                    CI, fn, constrainedAddresses,
                                    sampleOp.getSymbolAttr(), symbolTable)
                              : sampleOffset;
            FailureOr<SmallVector<Value>> read = readSiteValues(
                rewriter, sampleOp, isConstrained ? constraint : prevTrace,
                readOffset);
            if (failed(read))
              return failure();
            sampledValues = std::move(*read);
            if (flipIdx >= 0) {
              oldLive = oldLiveAt(loc, flipIdx);
              SmallVector<Value> drawn =
                  emitSiteDraw(rewriter, symbolTable, sampleOp);
              for (unsigned i = 1; i < sampledValues.size(); ++i)
                sampledValues[i] = impulse::SelectOp::create(
                    rewriter, loc, sampledValues[i].getType(), oldLive,
                    sampledValues[i], drawn[i]);
              sampledValues[0] = drawn[0];
            }
          }
          ArrayRef<Value> values = ArrayRef<Value>(sampledValues).drop_front();
          if (hastings && !deltaMode && (isSelected || flipIdx >= 0))
            hastingsTaint.insert(values.begin(), values.end());
          if (failed(verifySiteLogpdf(symbolTable, sampleOp, values.size())))
            return failure();

          bool termDirty =
              isSelected ||
              (deltaMode && llvm::any_of(siteParams(sampleOp), [&](Value v) {
                 return tainted.contains(v);
               }));
          if (deltaMode && isSelected)
            tainted.insert(values.begin(), values.end());

          Value oldScore;
          if (scoreTailSize > 0)
            oldScore = scalarAt(rewriter, loc, prevTrace,
                                {0, positionSize + siteOrdinal});

          if (scoring) {
            auto it = siteIndex.find(sampleOp.getSymbolAttr());
            if (it != siteIndex.end())
              weightAccumulator = recordSiteScore(
                  loc, weightAccumulator, it->second,
                  emitSiteLogpdf(rewriter, symbolTable, sampleOp, values));
          } else if (!deltaMode || termDirty) {
            Value logpdf =
                emitSiteLogpdf(rewriter, symbolTable, sampleOp, values);
            Value contribution =
                deltaMode
                    ? arith::SubFOp::create(rewriter, loc, logpdf, oldScore)
                          .getResult()
                    : logpdf;
            weightAccumulator = arith::AddFOp::create(
                rewriter, loc, weightAccumulator, contribution);
            if (scoreTailSize > 0)
              siteScores.push_back(logpdf);
            if (hastings && isSelected) {
              // A prior draw's proposal density cancels its own term.
              branch.corr = arith::AddFOp::create(rewriter, loc, branch.corr,
                                                  contribution);
              if (!deltaMode && !siteIndex.count(sampleOp.getSymbolAttr())) {
                // Parameters unchanged by the move: score the old value inline.
                Value oldLogpdf = emitSiteLogpdf(
                    rewriter, symbolTable, sampleOp,
                    sliceTraceValues(
                        rewriter, loc, prevTrace, sampleOffset,
                        TypeRange(sampleOp->getResultTypes()).drop_front()));
                branch.corr = arith::SubFOp::create(rewriter, loc, branch.corr,
                                                    oldLogpdf);
              }
            } else if (flipIdx >= 0) {
              // Kept: old density, cancelled via oldInfo; fresh: new density.
              Value term = arith::SelectOp::create(
                  rewriter, loc, oldLive,
                  scalarAt(rewriter, loc, oldInfo, {1, flipIdx}), logpdf);
              branch.corr =
                  arith::AddFOp::create(rewriter, loc, branch.corr, term);
            }
          } else {
            siteScores.push_back(oldScore);
          }
          ++siteOrdinal;

          if (addressInList(sampleOp, allAddresses) &&
              failed(appendSiteValues(rewriter, sampleOp, values, currTrace,
                                      currentTraceOffset,
                                      /*write=*/!deltaMode || isSelected)))
            return failure();
        } else {
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "Impulse: generative function body is empty; "
                "if this is a distribution, add a `logpdf_fn` attribute");
            return failure();
          }

          ArrayAttr subAllAddresses = buildSubSelection(
              rewriter, allAddresses, sampleOp.getSymbolAttr());
          ArrayAttr subSelection = buildSubSelection(
              rewriter, CI.getSelectionAttr(), sampleOp.getSymbolAttr());
          ArrayAttr subConstrainedAddrs =
              (constraint && constrainedAddresses)
                  ? buildSubSelection(rewriter, constrainedAddresses,
                                      sampleOp.getSymbolAttr())
                  : ArrayAttr();

          if (subAllAddresses.empty()) {
            sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subAllAddresses, symbolTable);
            if (subPositionSize <= 0) {
              sampleOp.emitError(
                  "Impulse: failed to compute sub-position size");
              return failure();
            }

            int64_t mergeOffset = computeOffsetForSampleInSelection(
                sampleOp, fn, allAddresses, sampleOp.getSymbolAttr(),
                symbolTable);
            if (mergeOffset < 0) {
              sampleOp.emitError("Impulse: failed to compute merge offset");
              return failure();
            }

            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            Value subPrevTrace = impulse::SliceOp::create(
                rewriter, sampleOp.getLoc(), subTraceType, prevTrace,
                rewriter.getDenseI64ArrayAttr({0, mergeOffset}),
                rewriter.getDenseI64ArrayAttr(
                    {1, mergeOffset + subPositionSize}),
                rewriter.getDenseI64ArrayAttr({1, 1}));

            Value subConstraint;
            ArrayAttr subConstrainedForOp;
            if (constraint && subConstrainedAddrs &&
                !subConstrainedAddrs.empty()) {
              int64_t subConstraintSize = computePositionSizeForSelection(
                  sampleOp, genFn, subConstrainedAddrs, symbolTable);
              int64_t subConstraintOffset = computeOffsetForSampleInSelection(
                  sampleOp, fn, constrainedAddresses, sampleOp.getSymbolAttr(),
                  symbolTable);
              if (subConstraintSize > 0 && subConstraintOffset >= 0) {
                auto subConstraintType = RankedTensorType::get(
                    {1, subConstraintSize}, rewriter.getF64Type());
                subConstraint = impulse::SliceOp::create(
                    rewriter, sampleOp.getLoc(), subConstraintType, constraint,
                    rewriter.getDenseI64ArrayAttr({0, subConstraintOffset}),
                    rewriter.getDenseI64ArrayAttr(
                        {1, subConstraintOffset + subConstraintSize}),
                    rewriter.getDenseI64ArrayAttr({1, 1}));
                subConstrainedForOp = subConstrainedAddrs;
              }
            }

            if (hastings) {
              bool genHasIf = false;
              genFn.getFunctionBody().walk(
                  [&](impulse::IfOp) { genHasIf = true; });
              bool inputsTainted = llvm::any_of(
                  llvm::drop_begin(sampleOp.getInputs()),
                  [&](Value v) { return hastingsTaint.contains(v); });
              if (inFlippableBranch(sampleOp) ||
                  (inputsTainted && (!subSelection.empty() || genHasIf))) {
                sampleOp.emitError(
                    "Impulse: MH over a generative call whose sites can "
                    "change liveness or parameters under the move is not "
                    "supported");
                return failure();
              }
            }

            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> regenResultTypes;
            regenResultTypes.push_back(subTraceType);
            regenResultTypes.push_back(
                hastings ? RankedTensorType::get({2}, rewriter.getF64Type())
                         : scalarTy);
            for (auto t : genFn.getResultTypes())
              regenResultTypes.push_back(t);

            auto nestedRegenerate = impulse::RegenerateOp::create(
                rewriter, sampleOp.getLoc(), regenResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subPrevTrace,
                subConstraint, subAllAddresses, subSelection,
                subConstrainedForOp, rewriter.getStringAttr(""));

            Value subTrace = nestedRegenerate.getNewTrace();
            Value subWeight = nestedRegenerate.getWeight();

            if (hastings) {
              nestedRegenerate->setAttr(kMHHastingsAttr,
                                        rewriter.getUnitAttr());
              auto [w, c] =
                  unpackHastingsWeight(rewriter, sampleOp.getLoc(), subWeight);
              weightAccumulator = arith::AddFOp::create(
                  rewriter, sampleOp.getLoc(), weightAccumulator, w);
              branch.corr = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                  branch.corr, c);
            } else if (!scoring) {
              weightAccumulator = arith::AddFOp::create(
                  rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);
            }

            currTrace = updateColumns(rewriter, sampleOp.getLoc(), currTrace,
                                      subTrace, mergeOffset);
            currentTraceOffset =
                std::max(currentTraceOffset, mergeOffset + subPositionSize);

            for (auto output : nestedRegenerate.getOutputs())
              sampledValues.push_back(output);
          }
        }

        rewriter.replaceAllUsesWith(sampleOp->getResults(), sampledValues);
        toErase.push_back(sampleOp);
        return success();
      };

      LogicalResult result = processImpulseBlock(
          NewF.getFunctionBody().front(), rewriter, carry, processSample);

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (failed(result)) {
        CI.emitError("Impulse: failed to walk sample ops");
        return failure();
      }

      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

        SmallVector<Value> newRetVals;
        Value outTrace =
            scoreTailSize > 0
                ? appendScoreTail(rewriter, retOp.getLoc(), carry.currTrace,
                                  siteScores, declaredWidth)
                : carry.currTrace;
        newRetVals.push_back(outTrace);
        if (hastings) {
          Value corr = carry.corr;
          if (oldInfo)
            for (int64_t k = 0, e = cast<RankedTensorType>(oldInfo.getType())
                                        .getShape()[1];
                 k < e; ++k)
              corr = arith::SubFOp::create(
                  rewriter, retOp.getLoc(), corr,
                  scalarAt(rewriter, retOp.getLoc(), oldInfo, {1, k}));
          newRetVals.push_back(
              packHastingsWeight(rewriter, retOp.getLoc(), carry.weight, corr));
        } else {
          newRetVals.push_back(carry.weight);
        }
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      if (hastings || scoring) {
        SmallVector<Type> resultTypes;
        NewF.walk([&](func::ReturnOp retOp) {
          resultTypes.assign(retOp.getOperandTypes().begin(),
                             retOp.getOperandTypes().end());
        });
        NewF.setType(FunctionType::get(CI.getContext(), NewF.getArgumentTypes(),
                                       resultTypes));
      }

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getOriginalTrace());
      if (hasConstraint)
        operands.push_back(CI.getConstraint());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      return success();
    }
  };
};

} // end anonymous namespace

void ExpandImpulsePass::runOnOperation() {
  // Must run before the per-op MH patterns lower these moves.
  MHRunBatcher::run(getOperation());

  RewritePatternSet patterns(&getContext());
  patterns.add<LowerUntracedCallPattern, LowerSimulatePattern,
               LowerGeneratePattern, LowerMHPattern, LowerRegeneratePattern>(
      &getContext());
  patterns.add<LowerMCMCPattern<impulse::InferOp>,
               LowerMCMCPattern<impulse::InferRegionOp>>(&getContext(),
                                                         debugDump);
  patterns.add<LowerVIPattern<impulse::InferOp>,
               LowerVIPattern<impulse::InferRegionOp>>(&getContext(),
                                                       debugDump);
  patterns.add<LowerMHInferPattern<impulse::InferOp>,
               LowerMHInferPattern<impulse::InferRegionOp>>(&getContext());
  patterns.add<LowerGibbsRegionPattern, RejectFlatGibbsPattern>(&getContext());

  mlir::GreedyRewriteConfig config;

  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
    signalPassFailure();
    return;
  }

  if (!postpasses.empty()) {
    mlir::PassManager pm(getOperation()->getContext());

    if (mlir::failed(mlir::parsePassPipeline(postpasses, pm))) {
      getOperation()->emitError()
          << "Failed to parse expand-impulse post-passes pipeline: "
          << postpasses;
      signalPassFailure();
      return;
    }

    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
      return;
    }
  }
}
